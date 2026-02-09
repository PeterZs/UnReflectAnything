import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from typing import Optional, Dict, List


class DINOv3_ConvNext(nn.Module):
    """
    Configurable DINOv3 ConvNeXt model with flexible return options.
    Supports returning hidden states and feature maps in various formats.
    ConvNeXt models output spatial feature maps directly (not token sequences).
    """

    def __init__(self, config):
        """
        Initialize DINOv3 ConvNeXt model with configuration.

        Args:
            config: dict containing:
                - model_name: str, DINOv3 ConvNeXt model name (default: "facebook/dinov3-convnext-large-pretrain-lvd1689m")
                - image_size: int, input image size (default: 896). The input image will be preprocessed to have this size (square)
                - freeze_backbone: bool, whether to freeze DINOv3 parameters (default: True)
                - return_last_hidden_state: bool, return last hidden state (default: True)
                - return_all_hidden_states: bool, return all hidden states (default: False)
                - return_selected_layers: list[int], specific layer indices to return (default: None)
                - return_as_feature_maps: bool, return as spatial feature maps (default: True for ConvNeXt)
                - return_cls_token: bool, return CLS token (default: False). For ConvNeXt, uses global average pooling
                - return_register_tokens: bool, return register tokens (default: False). Not available for ConvNeXt, returns empty tensor
        """
        super().__init__()

        # Configuration with defaults
        self.config = {
            "model_name": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "image_size": 896,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": [1, 2, 3, 4],
            "return_patch_tokens_only": True,
            "return_as_feature_maps": True,  # Default True for ConvNeXt (native format)
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,  # Override defaults with user config
        }
        # Override defaults with user config
        self.config["return_selected_layers"] = [1, 2, 3, 4]
        # DINOv3 ConvNeXt backbone
        self.dinov3 = AutoModel.from_pretrained(self.config["model_name"])
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
        self.processor.size = {
            "height": self.config["image_size"],
            "width": self.config["image_size"],
        }

        # Freeze parameters if requested
        if self.config["freeze_backbone"]:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        # Model properties - ConvNeXt outputs spatial feature maps
        # Get feature dimension from model config
        self.feature_dim = getattr(self.dinov3.config, "hidden_size", None) or getattr(
            self.dinov3.config, "embed_dim", None
        )
        if self.feature_dim is None:
            # Fallback: try to infer from model structure
            # ConvNeXt models typically have hidden_sizes in config
            if hasattr(self.dinov3.config, "hidden_sizes"):
                self.feature_dim = self.dinov3.config.hidden_sizes[-1]
            else:
                raise ValueError(
                    "Could not determine feature_dim from model config. Please specify manually."
                )

        # ConvNeXt uses patch size similar to ViT, typically 4 or 16
        # DINOv3 ConvNeXt uses patch_size=4 based on the architecture
        self.patch_size = getattr(self.dinov3.config, "patch_size", 4)
        self.dinov3.config.image_size = self.config["image_size"]

    def get_patch_spatial_dims(self, input_height, input_width):
        """Calculate spatial dimensions of feature maps based on input size."""
        # ConvNeXt typically downsamples by a factor related to patch_size
        # For DINOv3 ConvNeXt, the final feature map is typically H/patch_size x W/patch_size
        patch_h = input_height // self.patch_size
        patch_w = input_width // self.patch_size
        return patch_h, patch_w

    def feature_maps_to_tokens(self, feature_maps):
        """
        Convert spatial feature maps to token sequence format for API compatibility.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, N, C] - Token-like sequence format
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - convert to tokens
            B, C, H, W = feature_maps.shape
            # Flatten spatial dimensions and transpose to [B, N, C]
            tokens = feature_maps.flatten(2).transpose(1, 2)  # [B, H*W, C]
            return tokens
        elif len(feature_maps.shape) == 3:
            # Input is already [B, N, C] - return as-is
            return feature_maps
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def extract_cls_token(self, feature_maps):
        """
        Extract CLS-like token from feature maps using global average pooling.
        ConvNeXt doesn't have CLS tokens, so we use global average pooling.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, C] - Global pooled features (CLS-like token)
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - use global average pooling
            return F.adaptive_avg_pool2d(feature_maps, (1, 1)).squeeze(-1).squeeze(-1)
        elif len(feature_maps.shape) == 3:
            # Input is [B, N, C] - use mean pooling over sequence dimension
            return feature_maps.mean(dim=1)  # [B, C]
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def extract_register_tokens(self, feature_maps):
        """
        Extract register-like tokens from feature maps.
        ConvNeXt doesn't have register tokens, so we sample 4 corner locations or first 4 tokens.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, 4, C] - Sampled corner features (for API compatibility)
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - sample 4 corner locations
            B, C, H, W = feature_maps.shape
            # Vectorized sampling of 4 corner locations: top-left, top-right, bottom-left, bottom-right
            corners = torch.stack(
                [
                    feature_maps[:, :, 0, 0],  # top-left
                    feature_maps[:, :, 0, W - 1],  # top-right
                    feature_maps[:, :, H - 1, 0],  # bottom-left
                    feature_maps[:, :, H - 1, W - 1],  # bottom-right
                ],
                dim=1,
            )  # [B, 4, C]
            return corners
        elif len(feature_maps.shape) == 3:
            # Input is [B, N, C] - take first 4 tokens or pad if needed
            B, N, C = feature_maps.shape
            if N >= 4:
                return feature_maps[:, :4, :]  # [B, 4, C]
            else:
                # Pad with zeros if we have fewer than 4 tokens
                padding = torch.zeros(
                    B, 4 - N, C, device=feature_maps.device, dtype=feature_maps.dtype
                )
                return torch.cat([feature_maps, padding], dim=1)  # [B, 4, C]
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def forward(self, rgb_image):
        """
        Forward pass with configurable return options.

        Args:
            rgb_image: [B, 3, H, W] - Input RGB images (should be preprocessed for DINOv3)
                      Can be any size as long as H and W are divisible by patch_size

        Returns:
            dict containing requested outputs based on config:
                - 'last_hidden_state': [B, C, H', W'] (if return_as_feature_maps=True)
                                      or [B, H'*W', C] (if return_as_feature_maps=False)
                - 'cls_token': [B, C] - Global pooled features (CLS-like token)
                - 'register_tokens': [B, 4, C] - Sampled corner features (for compatibility)
                - 'all_hidden_states': List of feature maps or token sequences
                - 'selected_hidden_states': List of selected layer outputs
        """
        batch_size, _, input_h, input_w = rgb_image.shape

        # Ensure input dimensions are compatible with patch size
        assert input_h % self.patch_size == 0, (
            f"Height {input_h} must be divisible by patch size {self.patch_size}"
        )
        assert input_w % self.patch_size == 0, (
            f"Width {input_w} must be divisible by patch size {self.patch_size}"
        )

        # Calculate expected spatial dimensions
        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)

        # Get DINOv3 ConvNeXt outputs
        need_all_hidden_states = (
            self.config["return_all_hidden_states"]
            or self.config["return_selected_layers"] is not None
        )

        outputs = self.dinov3(rgb_image, output_hidden_states=need_all_hidden_states)

        # ConvNeXt outputs are spatial feature maps: [B, C, H', W']
        # The actual spatial dimensions may differ from patch_h/patch_w due to downsampling
        # We'll use the actual output dimensions

        # Prepare return dictionary
        result = {}

        # Get last hidden state - could be [B, C, H', W'] or [B, N, C] depending on model
        if (
            not hasattr(outputs, "last_hidden_state")
            or outputs.last_hidden_state is None
        ):
            raise ValueError(
                "Model did not return last_hidden_state. Check model output structure."
            )
        last_hidden = outputs.last_hidden_state

        # Detect the format of the output
        is_spatial = len(last_hidden.shape) == 4  # [B, C, H, W]
        is_tokens = len(last_hidden.shape) == 3  # [B, N, C]

        # Return last hidden state
        if self.config["return_last_hidden_state"]:
            if self.config["return_as_feature_maps"]:
                if is_spatial:
                    result["last_hidden_state"] = last_hidden  # [B, C, H', W']
                elif is_tokens:
                    # Convert tokens to spatial format if needed
                    # This requires knowing the spatial dimensions, which we can infer
                    B, N, C = last_hidden.shape
                    # Try to infer spatial dimensions from input size
                    # For ConvNeXt, typically N = (H/patch_size) * (W/patch_size)
                    # We'll reshape assuming square patches
                    spatial_size = int(N**0.5)
                    if spatial_size * spatial_size == N:
                        # Perfect square - reshape to spatial
                        result["last_hidden_state"] = last_hidden.transpose(1, 2).view(
                            B, C, spatial_size, spatial_size
                        )
                    else:
                        # Not a perfect square, keep as tokens but warn
                        result["last_hidden_state"] = last_hidden
                else:
                    result["last_hidden_state"] = last_hidden
            else:
                # Convert to token-like format for compatibility
                result["last_hidden_state"] = self.feature_maps_to_tokens(
                    last_hidden
                )  # [B, N, C]

        # Return CLS token (global average pooling)
        if self.config["return_cls_token"]:
            cls_token = self.extract_cls_token(last_hidden)  # [B, C]
            result["cls_token"] = cls_token

        # Return register tokens (corner sampling)
        if self.config["return_register_tokens"]:
            register_tokens = self.extract_register_tokens(last_hidden)  # [B, 4, C]
            result["register_tokens"] = register_tokens

        # Return all hidden states
        if self.config["return_all_hidden_states"]:
            all_hidden = outputs.hidden_states  # Tuple of [B, C, H', W'] or [B, N, C]
            if all_hidden is None:
                raise ValueError(
                    "Model did not return hidden_states. Make sure output_hidden_states=True "
                    "is passed to the model forward call."
                )
            # Convert tuple to list if needed
            if isinstance(all_hidden, tuple):
                all_hidden = list(all_hidden)
            if self.config["return_as_feature_maps"]:
                # Convert tokens to spatial maps if needed
                converted_hidden = []
                for h in all_hidden:
                    if len(h.shape) == 3:
                        # It's tokens [B, N, C], convert to spatial
                        B, N, C = h.shape
                        # Infer spatial dimensions assuming square patches
                        spatial_size = int(N**0.5)
                        if spatial_size * spatial_size == N:
                            # Perfect square - reshape to spatial
                            h_spatial = h.transpose(1, 2).view(
                                B, C, spatial_size, spatial_size
                            )
                            converted_hidden.append(h_spatial)
                        else:
                            # Not a perfect square, use patch dimensions
                            h_spatial = h.transpose(1, 2).view(B, C, patch_h, patch_w)
                            converted_hidden.append(h_spatial)
                    else:
                        # Already spatial [B, C, H, W]
                        converted_hidden.append(h)
                result["all_hidden_states"] = converted_hidden
            else:
                # Convert each to token-like format (or keep as tokens)
                result["all_hidden_states"] = [
                    self.feature_maps_to_tokens(h) for h in all_hidden
                ]

        # Return selected hidden states
        if self.config["return_selected_layers"] is not None:
            selected_layers = self.config["return_selected_layers"]
            if not hasattr(outputs, "hidden_states"):
                raise AttributeError(
                    f"Model outputs do not have 'hidden_states' attribute. "
                    f"Available attributes: {dir(outputs)}"
                )
            all_hidden = outputs.hidden_states
            if all_hidden is None:
                raise ValueError(
                    "Model did not return hidden_states. Make sure output_hidden_states=True "
                    "is passed to the model forward call."
                )
            # Convert tuple to list if needed
            if isinstance(all_hidden, tuple):
                all_hidden = list(all_hidden)
            elif not isinstance(all_hidden, (list, tuple)):
                raise TypeError(
                    f"Expected hidden_states to be a tuple or list, got {type(all_hidden)}"
                )
            # Check if we have any layers
            if len(all_hidden) == 0:
                raise ValueError(
                    "Model returned empty hidden_states. The model may not support "
                    "output_hidden_states or may have a different structure."
                )
            # Validate layer indices
            max_layer_idx = len(all_hidden) - 1
            invalid_layers = [i for i in selected_layers if i > max_layer_idx or i < 0]
            if invalid_layers:
                raise IndexError(
                    f"Selected layer indices {invalid_layers} are out of range. "
                    f"Model has {len(all_hidden)} layers (indices 0-{max_layer_idx})."
                )
            try:
                selected_hidden = [all_hidden[i] for i in selected_layers]
            except (IndexError, TypeError) as e:
                raise IndexError(
                    f"Failed to access hidden states at indices {selected_layers}. "
                    f"Model has {len(all_hidden)} layers. Error: {e}"
                ) from e
            if self.config["return_as_feature_maps"]:
                # Convert tokens to spatial maps if needed
                converted_hidden = []
                for h in selected_hidden:
                    if len(h.shape) == 3:
                        # It's tokens [B, N, C], convert to spatial
                        B, N, C = h.shape
                        # Infer spatial dimensions assuming square patches
                        spatial_size = int(N**0.5)
                        if spatial_size * spatial_size == N:
                            # Perfect square - reshape to spatial
                            h_spatial = h.transpose(1, 2).view(
                                B, C, spatial_size, spatial_size
                            )
                            converted_hidden.append(h_spatial)
                        else:
                            # Not a perfect square, use patch dimensions
                            h_spatial = h.transpose(1, 2).view(B, C, patch_h, patch_w)
                            converted_hidden.append(h_spatial)
                    else:
                        # Already spatial [B, C, H, W]
                        converted_hidden.append(h)
                result["selected_hidden_states"] = converted_hidden
            else:
                # Convert each to token-like format (or keep as tokens)
                result["selected_hidden_states"] = [
                    self.feature_maps_to_tokens(h) for h in selected_hidden
                ]

        return result

    def preprocess_image(self, image_tensor):
        """
        Preprocess image for DINOv3 ConvNeXt using the proper processor.

        Args:
            image_tensor: [B, 3, H, W] - Raw image tensor with values in [0, 1]

        Returns:
            [B, 3, H, W] - Preprocessed tensor ready for DINOv3 ConvNeXt
        """
        import PIL.Image

        # Vectorized conversion to [0, 255] range and permute dimensions
        # [B, 3, H, W] -> [B, H, W, 3]
        img_batch = (image_tensor * 255).byte().permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Convert entire batch to numpy for PIL processing
        img_numpy = img_batch.cpu().numpy()  # [B, H, W, 3]

        # Create PIL images from the entire batch
        pil_images = [
            PIL.Image.fromarray(img_numpy[i], mode="RGB")
            for i in range(img_numpy.shape[0])
        ]

        # Process entire batch at once using the processor
        processed = self.processor(images=pil_images, return_tensors="pt")
        processed_batch = processed["pixel_values"]  # [B, 3, H, W]

        return processed_batch.to(image_tensor.device)

class DPTReassembleLayer_ConvNext(nn.Module):
    """
    Reassemble layer for ConvNeXt feature maps (already spatial).
    Handles projection and upsampling/downsampling without token reshaping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        readout_type: str = "ignore",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.readout_type = readout_type

        # Readout projection if using "project" method
        if readout_type == "project":
            self.readout_project = nn.Sequential(
                nn.Conv2d(2 * in_channels, in_channels, kernel_size=1), nn.GELU()
            )

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Spatial resampling based on scale factor
        if scale_factor == 4.0:
            # 4x upsampling
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=True,
            )
        elif scale_factor == 2.0:
            # 2x upsampling
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        elif scale_factor == 1.0:
            # No resampling
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            # 0.5x downsampling
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] - Spatial feature maps from ConvNeXt (already spatial)
        Returns:
            [B, out_channels, H', W'] - Spatial feature map after projection and resampling
        """
        # Project channels
        feature_map = self.proj(feature_map)  # [B, out_channels, H, W]

        # Resample spatial resolution
        output = self.resample(feature_map)  # [B, out_channels, H', W']

        return output


class DPT_Decoder_ConvNext(nn.Module):
    """
    DPT decoder adapted for ConvNeXt feature maps.
    Works with spatial feature maps directly (no token reshaping needed).
    Implements multi-scale reassembly, progressive fusion, and RGB prediction head.

    Dropout:
        Controlled by config["dropout"] (float in [0,1]). Applied as 2D dropout
        after each fusion addition and between stages in the head to regularize
        training. Shapes are preserved; tensor dims are:
        - inputs hidden_states: List[4] of [B, C, H, W] (spatial feature maps)
        - output: [B, C_out, H, W]
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Default configuration
        default_config = {
            "feature_dim": 768,  # ConvNeXt feature dimension (can be int or list)
            "reassemble_out_channels": [96, 192, 384, 768],  # Neck hidden sizes
            "reassemble_factors": [4.0, 2.0, 1.0, 0.5],  # Spatial scale factors
            "fusion_hidden_size": 256,
            "readout_type": "ignore",  # 'ignore', 'add', or 'project'
            "use_bn": False,
            "dropout": 0.0,
            "output_image_size": (448, 448),  # If None, maintains input size
            "output_channels": 3,  # Set to 4 for RGBA output
        }

        self.config = {**default_config, **(config or {})}
        self.config["feature_dim"] = [192, 384, 768, 1536]
        self.out_image_size = self.config["output_image_size"]

        # Handle feature_dim: can be int (same for all) or list (per-stage)
        feature_dim = self.config["feature_dim"]
        if isinstance(feature_dim, int):
            # Use same feature_dim for all stages
            feature_dims = [feature_dim] * len(self.config["reassemble_out_channels"])
        elif isinstance(feature_dim, (list, tuple)):
            # Use per-stage feature dimensions
            if len(feature_dim) != len(self.config["reassemble_out_channels"]):
                raise ValueError(
                    f"feature_dim list length ({len(feature_dim)}) must match "
                    f"reassemble_out_channels length ({len(self.config['reassemble_out_channels'])})"
                )
            feature_dims = list(feature_dim)
        else:
            raise TypeError(
                f"feature_dim must be int or list/tuple, got {type(feature_dim)}"
            )

        # Create reassemble layers for multi-scale feature extraction
        self.reassemble_layers = nn.ModuleList(
            [
                DPTReassembleLayer_ConvNext(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    scale_factor=scale,
                    readout_type=self.config["readout_type"],
                )
                for in_ch, out_ch, scale in zip(
                    feature_dims,
                    self.config["reassemble_out_channels"],
                    self.config["reassemble_factors"],
                )
            ]
        )

        # Create fusion blocks for progressive feature combination
        fusion_in_channels = self.config["reassemble_out_channels"]
        self.fusion_blocks = nn.ModuleList(
            [
                DPTFeatureFusionBlock(
                    in_channels=ch,
                    out_channels=self.config["fusion_hidden_size"],
                    use_bn=self.config["use_bn"],
                )
                for ch in fusion_in_channels
            ]
        )

        # Dropout used after fusion additions and inside head stages
        p = float(self.config.get("dropout", 0.0))
        self.drop2d = nn.Dropout2d(p) if p and p > 0.0 else nn.Identity()

        # RGB prediction head - split into stages to avoid large intermediate tensors
        # This allows progressive upsampling to handle very large output sizes
        self.rgb_head_stage1 = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config["fusion_hidden_size"], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_stage2 = nn.Sequential(
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_stage3 = nn.Sequential(
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_final = nn.Sequential(
            # Final RGB projection
            nn.Conv2d(32, self.config["output_channels"], kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through DPT decoder for ConvNeXt.

        Args:
            hidden_states: List of 4 tensors from DINOv3_ConvNext selected layers
                          Each tensor: [B, C, H, W] (spatial feature maps)
                          Can have different channel dimensions per stage

        Returns:
            rgb_output: [B, 3, H, W] - RGB image in [0, 1] range
        """
        input_height, input_width = self.out_image_size

        # Validate input
        if len(hidden_states) != len(self.reassemble_layers):
            raise ValueError(
                f"Expected {len(self.reassemble_layers)} hidden states, "
                f"got {len(hidden_states)}"
            )

        # Apply reassemble layers to create multi-scale feature maps
        reassembled_features = []
        for i, (hidden_state, reassemble) in enumerate(
            zip(hidden_states, self.reassemble_layers)
        ):
            # Verify channel dimension matches
            actual_channels = hidden_state.shape[1]
            expected_channels = reassemble.in_channels
            if actual_channels != expected_channels:
                actual_dims = [h.shape[1] for h in hidden_states]
                raise ValueError(
                    f"Stage {i}: Expected {expected_channels} channels, "
                    f"got {actual_channels} channels. "
                    f"Hidden state shape: {hidden_state.shape}.\n\n"
                    f"To fix this, set 'feature_dim' in your decoder config to:\n"
                    f"  feature_dim: {actual_dims}\n\n"
                    f"Or in YAML format:\n"
                    f"  FEATURE_DIM: {actual_dims}\n\n"
                    f"Current decoder config feature_dim: {self.config.get('feature_dim', 'not set')}"
                )
            feature_map = reassemble(hidden_state)  # [B, out_ch, H', W']
            reassembled_features.append(feature_map)

        # Progressive fusion from smallest to largest scale
        # Use actual spatial dimensions from reassembled features to ensure matching

        # Start with smallest scale (stage 3)
        fused = self.fusion_blocks[3](reassembled_features[3])  # [B, 256, H3, W3]
        _h3, _w3 = fused.shape[2], fused.shape[3]

        # Interpolate to match stage 2 size
        h2, w2 = reassembled_features[2].shape[2], reassembled_features[2].shape[3]
        fused = F.interpolate(
            fused, size=(h2, w2), mode="bilinear", align_corners=True
        )  # [B, 256, H2, W2]

        # Add stage 2 features
        stage2_fused = self.fusion_blocks[2](
            reassembled_features[2]
        )  # [B, 256, H2, W2]
        # Ensure spatial dimensions match (should already match, but be safe)
        if fused.shape[2:] != stage2_fused.shape[2:]:
            stage2_fused = F.interpolate(
                stage2_fused,
                size=(fused.shape[2], fused.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        fused = fused + stage2_fused  # [B, 256, H2, W2]
        fused = self.drop2d(fused)

        # Interpolate to match stage 1 size
        h1, w1 = reassembled_features[1].shape[2], reassembled_features[1].shape[3]
        fused = F.interpolate(
            fused, size=(h1, w1), mode="bilinear", align_corners=True
        )  # [B, 256, H1, W1]

        # Add stage 1 features
        stage1_fused = self.fusion_blocks[1](
            reassembled_features[1]
        )  # [B, 256, H1, W1]
        # Ensure spatial dimensions match
        if fused.shape[2:] != stage1_fused.shape[2:]:
            stage1_fused = F.interpolate(
                stage1_fused,
                size=(fused.shape[2], fused.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        fused = fused + stage1_fused  # [B, 256, H1, W1]
        fused = self.drop2d(fused)

        # Interpolate to match stage 0 size
        h0, w0 = reassembled_features[0].shape[2], reassembled_features[0].shape[3]
        fused = F.interpolate(
            fused, size=(h0, w0), mode="bilinear", align_corners=True
        )  # [B, 256, H0, W0]

        # Add stage 0 features
        stage0_fused = self.fusion_blocks[0](
            reassembled_features[0]
        )  # [B, 256, H0, W0]
        # Ensure spatial dimensions match
        if fused.shape[2:] != stage0_fused.shape[2:]:
            stage0_fused = F.interpolate(
                stage0_fused,
                size=(fused.shape[2], fused.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        fused = fused + stage0_fused  # [B, 256, H0, W0]
        fused = self.drop2d(fused)

        # Apply RGB head with progressive upsampling to avoid large intermediate tensors
        # Calculate target size early to do progressive upsampling
        target_size = self.config["output_image_size"] or (input_height, input_width)
        target_h, target_w = target_size

        # Stage 1: Process at current resolution
        x = self.rgb_head_stage1(fused)  # [B, 128, H, W]

        # Progressive upsampling: process at lower resolution first, then upsample
        # This avoids creating huge intermediate tensors
        # Calculate safe intermediate size (max 896x896 to avoid INT_MAX issues with batch_size=32)
        max_safe_size = 896
        current_h, current_w = x.shape[2], x.shape[3]

        # If target is very large, process at intermediate size first
        if target_h > max_safe_size or target_w > max_safe_size:
            # Process stage 2 at current or intermediate size
            intermediate_h = min(max_safe_size, max(current_h * 2, target_h // 2))
            intermediate_w = min(max_safe_size, max(current_w * 2, target_w // 2))

            # Upsample to intermediate size if needed
            if x.shape[2:] != (intermediate_h, intermediate_w):
                x = F.interpolate(
                    x,
                    size=(intermediate_h, intermediate_w),
                    mode="bilinear",
                    align_corners=True,
                )

            # Stage 2: Process at intermediate size
            x = self.rgb_head_stage2(x)  # [B, 64, H_intermediate, W_intermediate]

            # Stage 3: Process at intermediate size
            x = self.rgb_head_stage3(x)  # [B, 32, H_intermediate, W_intermediate]

            # Final RGB projection at intermediate size
            x = self.rgb_head_final(x)  # [B, 3, H_intermediate, W_intermediate]

            # Final upsample to target size
            rgb_output = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=True
            )
        else:
            # Target size is manageable, process normally
            # Upsample to target size after stage 1
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(
                    x, size=(target_h, target_w), mode="bilinear", align_corners=True
                )

            # Stage 2: Process
            x = self.rgb_head_stage2(x)  # [B, 64, H_target, W_target]

            # Stage 3: Process
            x = self.rgb_head_stage3(x)  # [B, 32, H_target, W_target]

            # Final RGB projection
            rgb_output = self.rgb_head_final(x)  # [B, 3, H_target, W_target]

        return rgb_output

# ---- 1) A FiLM-enabled DPT that can be used for the diffuse/spec decoders ----
class FiLMConditionedDPT(DPT_Decoder):
    """
    Drop-in replacement for DPT_Decoder that accepts a spatial mask (B,1,H,W)
    (e.g., predicted highlight map) and applies FiLM at each fusion stage.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._mask = None
        self._mask_enabled = False

        # For each fusion stage we predict [gamma, beta] from [mask, distance]
        self.film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, self.config["fusion_hidden_size"] * 2, 1),
                )
                for _ in range(4)
            ]
        )

    @torch.no_grad()
    def _build_mask_pyr(self, m: torch.Tensor):
        """
        m: (B,1,H,W) in [0,1]. Produces a pyramid of [mask, distance] at stage sizes.
        Stage spatial sizes after fusion are: 12, 24, 48, 96 (before last upsample).
        """
        H, W = self.out_image_size
        # Cheap distance proxy (replace with proper EDT if you have it)
        inv = 1.0 - m
        d = inv
        for _ in range(3):
            d = F.avg_pool2d(d, 3, 1, 1)
        d = 1.0 - d
        d = d / (d.amax(dim=(-1, -2), keepdim=True) + 1e-6)

        # Spatial sizes to match fused feature maps at each fusion stage:
        # stage3 -> (H/32, W/32), stage2 -> (H/16, W/16), stage1 -> (H/8, W/8), stage0 -> (H/4, W/4)
        sizes = [
            (H // 32, W // 32),  # e.g., 14x14 when H=W=448 (12x12 when H=W=384)
            (H // 16, W // 16),  # e.g., 28x28 (24x24)
            (H // 8, W // 8),  # e.g., 56x56 (48x48)
            (H // 4, W // 4),  # e.g., 112x112 (96x96)
            # (H // 2,  W // 2),   # e.g., 112x112 (96x96)
        ]

        pyr = []
        for hh, ww in sizes:
            mm = F.interpolate(m, size=(hh, ww), mode="nearest")
            dd = F.interpolate(d, size=(hh, ww), mode="bilinear", align_corners=True)
            pyr.append(torch.cat([mm, dd], dim=1))  # (B,2,hh,ww)
        return pyr

    def set_mask(self, mask: torch.Tensor | None):
        """
        mask: (B,1,H,W) in [0,1] (soft is fine). Set to None to disable FiLM.
        """
        self._mask = mask
        self._mask_enabled = mask is not None

    def _apply_film(self, x: torch.Tensor, ab_layer: nn.Module, cond: torch.Tensor):
        ab = ab_layer(cond)  # (B, 2C, H, W)
        gamma, beta = torch.chunk(ab, 2, dim=1)
        return gamma * x + beta

    def forward(self, hidden_states, return_mask: bool = False):
        # Standard reassembly part (unchanged)
        H, W = self.out_image_size
        ph, pw = H // 16, W // 16

        feats = []
        for hs, reas in zip(hidden_states, self.reassemble_layers):
            feats.append(
                reas(hs, ph, pw)
            )  # sizes: [96,96], [48,48], [24,24], [12,12] in channels [96,192,384,768]

        # Prepare mask pyramid if available
        mask_pyr = (
            self._build_mask_pyr(self._mask) if self._mask_enabled else [None] * 4
        )

        # Fusion with FiLM at every stage
        fused = self.fusion_blocks[3](feats[3])  # (B,256,12,12)
        if mask_pyr[0] is not None:
            fused = self._apply_film(fused, self.film[3], mask_pyr[0])

        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 24

        fused = fused + self.fusion_blocks[2](feats[2])  # (B,256,24,24)
        if mask_pyr[1] is not None:
            fused = self._apply_film(fused, self.film[2], mask_pyr[1])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 48

        fused = fused + self.fusion_blocks[1](feats[1])  # (B,256,48,48)
        if mask_pyr[2] is not None:
            fused = self._apply_film(fused, self.film[1], mask_pyr[2])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 96

        fused = fused + self.fusion_blocks[0](feats[0])  # (B,256,96,96)
        if mask_pyr[3] is not None:
            fused = self._apply_film(fused, self.film[0], mask_pyr[3])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 192

        out = self.rgb_head(fused)  # (B,C,H,W) after final resize
        if self.config["output_image_size"] or (out.shape[-2:] != (H, W)):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        if return_mask:
            return out, mask_pyr
        return out


# ---- 2) Wiring in RGBDistillDecomposer: run highlight first, then condition diffuse/spec ----
class UnReflect_Model_FiLMConditioned(UnReflect_Model):
    """
    Assumes decoders dict contains at least:
      - "highlight": DPT_Decoder(output_channels=1) that predicts soft mask in [0,1]
      - "diffuse":   FiLMConditionedDPT(...)        that will be conditioned by the mask
    You can add "specular" similarly if desired.
    """

    def forward(self, model_input_dict):
        x = model_input_dict["rgb"]  # (B,3,H,W) in [0,1]
        rgb_in = self.dinov3.preprocess_image(x)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ]  # List[4] of [B,N_p,C]

        outputs = {}

        # 1) Predict highlight mask (soft), from highlight head
        if "highlight" not in self.decoders:
            raise KeyError("decoders must include a 'highlight' head for Option A.")
        hl_logits = self.decoders["highlight"](
            tokens_list
        )  # (B,1,H,W) in [0,1] (Sigmoid in head)
        mask_soft = hl_logits.clamp(
            0, 1
        )  # treat as soft attention; no hard threshold here
        outputs["highlight"] = mask_soft

        # 2) Condition other heads with FiLM if supported
        for name, dec in self.decoders.items():
            if name == "highlight":
                continue
            if hasattr(dec, "set_mask"):
                dec.set_mask(mask_soft)
            outputs[name] = dec(tokens_list)
            # outputs["mask_pyr"] = mask_pyr
        return outputs

    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
        return tokens_list
