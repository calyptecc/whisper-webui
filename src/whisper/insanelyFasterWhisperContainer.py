import json

import argparse
import os
from typing import List
import huggingface_hub
import torch
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from transformers.pipelines import Pipeline

from src.config import ModelConfig
from src.hooks.progressListener import ProgressListener
from src.modelCache import GLOBAL_MODEL_CACHE, ModelCache
from src.prompts.abstractPromptStrategy import AbstractPromptStrategy
from src.utils import format_timestamp
from src.whisper.abstractWhisperContainer import AbstractWhisperCallback, AbstractWhisperContainer

class InsanelyFasterWhisperContainer(AbstractWhisperContainer):
    def __init__(self, model_name: str, device: str = None, 
                       compute_type: str = "float16", flash: bool = False,
                       download_root: str = None,
                       cache: ModelCache = None, models: List[ModelConfig] = []):
        super().__init__(model_name, device, compute_type, download_root, cache, models)
        self.flash = flash
    
    def ensure_downloaded(self):
        """
        Ensure that the model is downloaded. This is useful if you want to ensure that the model is downloaded before
        passing the container to a subprocess.
        """
        model_config = self._get_model_config()
        model_url = self._get_model_url()
        
        if os.path.isdir(model_config.url):
            model_config.path = model_config.url
        else:
            kwargs = {}
            
            if self.download_root is not None:
                kwargs["local_dir"] = self.download_root
            model_config.path = huggingface_hub.snapshot_download(model_url, **kwargs)

    def _get_model_config(self) -> ModelConfig:
        """
        Get the model configuration for the model.
        """
        for model in self.models:
            if model.name == self.model_name:
                return model
        return None

    def _get_model_url(self) -> str:
        """
        Get the model URL for the model.
        """
        model_config = self._get_model_config()
        model_url = model_config.url

        if model_config.type == "whisper":
            if model_url in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
                return "openai/whisper-" + model_url
            else:
                raise Exception("InsanelyFasterWhisperContainer does not yet support Whisper models.")
        return model_config.url

    def _create_model(self) -> Pipeline:
        print("Loading insanely faster whisper model " + self.model_name + " for device " + str(self.device))
        model_url = self._get_model_url()
        device = self.device

        if (device is None):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if self.compute_type == "float16" else torch.float32

        if self.flash == True:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_url,
                torch_dtype=dtype,
                device=device,
                model_kwargs={"use_flash_attention_2": True},
            )
        else:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_url,
                torch_dtype=dtype,
                device=device,
                )

            pipe.model = pipe.model.to_bettertransformer()

        pipe.binary_output
        return pipe

    def create_callback(self, language: str = None, task: str = None, 
                        prompt_strategy: AbstractPromptStrategy = None, 
                        **decodeOptions: dict) -> AbstractWhisperCallback:
        """
        Create a WhisperCallback object that can be used to transcript audio files.

        Parameters
        ----------
        language: str
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        prompt_strategy: AbstractPromptStrategy
            The prompt strategy to use. If not specified, the prompt from Whisper will be used.
        decodeOptions: dict
            Additional options to pass to the decoder. Must be pickleable.

        Returns
        -------
        A WhisperCallback object.
        """
        return InsanelyFasterWhisperCallback(self, language=language, task=task, prompt_strategy=prompt_strategy, **decodeOptions)

    # This is required for multiprocessing
    def __getstate__(self):
        return { 
            "model_name": self.model_name, 
            "device": self.device, 
            "download_root": self.download_root, 
            "models": self.models, 
            "compute_type": self.compute_type,
            "flash": self.flash
        }

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self.device = state["device"]
        self.download_root = state["download_root"]
        self.models = state["models"]
        self.compute_type = state["compute_type"]
        self.flash = state["flash"]
        self.model = None
        # Depickled objects must use the global cache
        self.cache = GLOBAL_MODEL_CACHE

class InsanelyFasterWhisperCallback(AbstractWhisperCallback):
    def __init__(self, model_container: InsanelyFasterWhisperContainer, language: str = None, task: str = None, 
                 prompt_strategy: AbstractPromptStrategy = None, 
                 **decodeOptions: dict):
        self.model_container = model_container
        self.language = language
        self.task = task
        self.prompt_strategy = prompt_strategy
        self.decodeOptions = decodeOptions

        self._printed_warning = False
        
    def invoke(self, audio, segment_index: int, prompt: str, detected_language: str, progress_listener: ProgressListener = None):
        """
        Peform the transcription of the given audio file or data.

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor]
            The audio file to transcribe, or the audio data as a numpy array or torch tensor.
        segment_index: int
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        progress_listener: ProgressListener
            A callback to receive progress updates.
        """
        model = self.model_container.get_model()
        
        # Copy decode options and remove options that are not supported by faster-whisper
        decodeOptions = self.decodeOptions.copy()
        verbose = decodeOptions.pop("verbose", None)

        # Not supported
        decodeOptions.pop("word_timestamps", None)
        decodeOptions.pop("initial_prompt", None)

        # Pass to the model
        batch_size = decodeOptions.pop("batch_size", 24)
        ts = decodeOptions.pop("return_timestamps", True)

        #initial_prompt = self.prompt_strategy.get_segment_prompt(segment_index, prompt, detected_language) \
        #                 if self.prompt_strategy else prompt
        
        #if initial_prompt is not None:
        #    print("Initial prompt is not supported by insanely faster whisper.")

        outputs = model(
            audio,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=decodeOptions,
            return_timestamps=ts,
        )

        segments = []

        # Format: 
        #   { 
        #       "text": "Hello world",
        #       "chunks": [
        #           { "text": "Hello", "timestamp": [1, 10] },
        #       ]
        #   }
        if outputs is not None and outputs["chunks"] is not None:
            for chunk in outputs["chunks"]:
                chunk_text = chunk["text"]
                # [ start, end ]
                chunk_timestamps = chunk["timestamp"]
                chunk_start = chunk_timestamps[0]
                chunk_end = chunk_timestamps[1]

                segments.append({
                    "text": chunk_text,
                    "start": chunk_start,
                    "end": chunk_end
                })

                if verbose:
                    print("[{}->{}] {}".format(format_timestamp(chunk_start, True), format_timestamp(chunk_end, True), chunk_text))

        text = outputs["text"]

        result = {
            "segments": segments,
            "text": text,
            "language": outputs.get("language", None) or detected_language,
        }

        # If we have a prompt strategy, we need to increment the current prompt
        if self.prompt_strategy:
            self.prompt_strategy.on_segment_finished(segment_index, prompt, detected_language, result)

        if progress_listener is not None:
            progress_listener.on_finished()
        return result

def main():
    parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
    parser.add_argument(
        "--file-name",
        required=True,
        type=str,
        help="Path or URL to the audio file to be transcribed.",
    )
    parser.add_argument(
        "--device-id",
        required=False,
        default="0",
        type=str,
        help='Device ID for your GPU (just pass the device ID number). (default: "0")',
    )
    parser.add_argument(
        "--transcript-path",
        required=False,
        default="output.json",
        type=str,
        help="Path to save the transcription output. (default: output.json)",
    )
    parser.add_argument(
        "--model-name",
        required=False,
        default="openai/whisper-large-v3",
        type=str,
        help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--task",
        required=False,
        default="transcribe",
        type=str,
        choices=["transcribe", "translate"],
        help="Task to perform: transcribe or translate to another language. (default: transcribe)",
    )
    parser.add_argument(
        "--language",
        required=False,
        type=str,
        default="None",
        help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=24,
        help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
    )
    parser.add_argument(
        "--flash",
        required=False,
        type=bool,
        default=False,
        help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
    )
    parser.add_argument(
        "--timestamp",
        required=False,
        type=str,
        default="chunk",
        choices=["chunk", "word"],
        help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
    )

    args = parser.parse_args()

    if args.flash == True:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            device=f"cuda:{args.device_id}",
            model_kwargs={"use_flash_attention_2": True},
        )
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            device=f"cuda:{args.device_id}",
        )

        pipe.model = pipe.model.to_bettertransformer()

    if args.timestamp == "word":
        ts = "word"

    else:
        ts = True

    if args.language == "None":
        lang = None

    else:
        lang = args.language

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        outputs = pipe(
            args.file_name,
            chunk_length_s=30,
            batch_size=args.batch_size,
            generate_kwargs={"task": args.task, "language": lang},
            return_timestamps=ts,
        )

    with open(args.transcript_path, "w", encoding="utf8") as fp:
        json.dump(outputs, fp, ensure_ascii=False)

    print(
        f"Voila! Your file has been transcribed go check it out over here! {args.transcript_path}"
    )

if __name__ == "__main__":
    main()