from pegasus import Pegasus
from next.mm_encoders import load_and_transform_video_data
from next.transformer import ViTransformerWrapper, Encoder


#encoders
class AudioEncoder(Pegasus):
    # audio_encoder = AudioEncoder()
    # audio_embeddings = audio_encoder.embed_audio_data([audio1, audio2])  # You'd provide your list of audio data here.
    def __init__(
        self,
        multi_process=False,
        n_processors=1,
        hosted=False
    ):
        super().__init__(
            "audio",
            multi_process,
            n_processors,
            hosted
        )
    def embed(self, audio):
        return self.embed_data(audio)
    
class VideoEncoder(Pegasus):
    """
    from next import VideoEncoder

    device = torch.device(
        "cuda" if torch.cuda.is_available()
    )

    video_encoder = VideoEncoder()
    video_embeddings = video_encoder.embed([video, video2], device)
    
    """
    def __init__(
        self,
        multi_process=False,
        n_processors=1,
        hosted=False
    ):
        super().__init__(
            "vision", 
            multi_process, 
            n_processors,
            hosted
        )
    
    def embed(self, video, device):
        video = load_and_transform_video_data(video, device)
        return self.embed_data(video)

class ImageEncoder:
    # # Usage:
    # image_encoder = ImageEncoder()
    # img_embeddings = image_encoder.embed_image_data([img1, img2])  # You'd provide your list of image data here.

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
    ):
        super().__init__()
        
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
            )
        )

    def embed(self, img):
        encoded = self.encoder(img, return_embeddings=True)
        return encoded
