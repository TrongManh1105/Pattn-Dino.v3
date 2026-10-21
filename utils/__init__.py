class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride

        # Tính số lượng patch
        self.patch_num = int((self.seq_len - self.patch_size) / self.stride + 1)

        # =========================================================
        # THAY ENCODER / BACKBONE BẰNG DINOv3
        # =========================================================
        self.encoder = AutoModel.from_pretrained("facebook/dinov3-base-p16")
        self.d_model = self.encoder.config.hidden_size  # Tự động lấy 768

        # Freeze weights của DINOv3 để làm Feature Extractor
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Cập nhật Lớp Projection & Head phù hợp d_model mới
        self.input_projection = nn.Linear(self.patch_size, self.d_model)
        self.head = nn.Linear(self.patch_num * self.d_model, self.pred_len)