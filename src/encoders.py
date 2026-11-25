from PIL import Image
from sentence_transformers import SentenceTransformer

from src.config import BATCH_SIZE, IMAGE_MODEL_NAME, TEXT_MODEL_NAME


class MultimodalEncoder:
    def __init__(
        self,
        image_model_name: str = IMAGE_MODEL_NAME,
        text_model_name: str = TEXT_MODEL_NAME,
    ) -> None:
        self.image_model_name = image_model_name
        self.text_model_name = text_model_name
        self._image_encoder: SentenceTransformer | None = None
        self._text_encoder: SentenceTransformer | None = None

    @property
    def image_encoder(self) -> SentenceTransformer:
        if self._image_encoder is None:
            self._image_encoder = SentenceTransformer(self.image_model_name)
        return self._image_encoder

    @property
    def text_encoder(self) -> SentenceTransformer:
        if self._text_encoder is None:
            self._text_encoder = SentenceTransformer(self.text_model_name)
        return self._text_encoder

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        embeddings = self.image_encoder.encode(
            images,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_text(self, query: str) -> list[float]:
        embedding = self.text_encoder.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        return embedding.tolist()