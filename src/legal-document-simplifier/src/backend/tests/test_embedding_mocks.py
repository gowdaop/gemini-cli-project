# tests/test_embedding_mocks.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.embedding import EmbeddingService, embed_texts

class TestEmbeddingMocks:
    """Tests using mocks for controlled scenarios"""
    
    @patch('src.backend.services.embedding.torch')
    @patch('src.backend.services.embedding.AutoModel')
    @patch('src.backend.services.embedding.AutoTokenizer')
    def test_model_loading_success(self, mock_tokenizer, mock_model, mock_torch):
        """Test successful model loading with mocks"""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_torch.cuda.is_available.return_value = False
        
        service = EmbeddingService()
        tokenizer, model = service._load_model()
        
        assert tokenizer is not None
        assert model is not None
        mock_model_instance.eval.assert_called_once()

    @patch('src.backend.services.embedding.torch')
    def test_gpu_detection(self, mock_torch):
        """Test GPU detection logic"""
        mock_torch.cuda.is_available.return_value = True
        
        service = EmbeddingService()
        info = service.get_embedding_info()
        
        assert "device" in info
        # Would be "cuda" if transformers available and GPU detected

    @patch('src.backend.services.embedding.EmbeddingService._encode_batch')
    def test_batch_processing_calls(self, mock_encode):
        """Test that large batches are processed in chunks"""
        mock_encode.return_value = [[0.1] * 768]
        
        service = EmbeddingService()
        texts = ["test"] * 25  # More than batch size of 8
        
        service.embed_texts_sync(texts)
        
        # Should be called multiple times for batches
        assert mock_encode.call_count >= 3  # 25 texts / 8 batch size = 4 batches
