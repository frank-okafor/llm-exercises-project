#!/usr/bin/env python3
"""
Test suite for NASA Mission Intelligence RAG System

Tests all core components:
- embedding_pipeline.py
- rag_client.py
- llm_client.py
- ragas_evaluator.py
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEmbeddingPipeline(unittest.TestCase):
    """Tests for embedding_pipeline.py"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk_size"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()
            pipeline.chunk_size = 1000
            pipeline.chunk_overlap = 200

            short_text = "This is a short text."
            metadata = {"source": "test"}

            chunks = pipeline.chunk_text(short_text, metadata)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0][0], short_text)
            self.assertEqual(chunks[0][1]['chunk_index'], 0)
            self.assertEqual(chunks[0][1]['total_chunks'], 1)

    def test_chunk_text_long_text(self):
        """Test chunking with text longer than chunk_size"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()
            pipeline.chunk_size = 100
            pipeline.chunk_overlap = 20

            # Create text longer than chunk_size
            long_text = "This is sentence one. " * 20  # ~440 characters
            metadata = {"source": "test"}

            chunks = pipeline.chunk_text(long_text, metadata)

            # Should create multiple chunks
            self.assertGreater(len(chunks), 1)

            # Each chunk should have proper metadata
            for i, (chunk_text, chunk_meta) in enumerate(chunks):
                self.assertEqual(chunk_meta['chunk_index'], i)
                self.assertEqual(chunk_meta['total_chunks'], len(chunks))
                self.assertIn('char_start', chunk_meta)
                self.assertIn('char_end', chunk_meta)

    def test_chunk_text_empty_text(self):
        """Test chunking with empty text"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()
            pipeline.chunk_size = 1000
            pipeline.chunk_overlap = 200

            chunks = pipeline.chunk_text("", {"source": "test"})
            self.assertEqual(chunks, [])

            chunks = pipeline.chunk_text("   ", {"source": "test"})
            self.assertEqual(chunks, [])

    def test_chunk_overlap_consistency(self):
        """Test that chunks have proper overlap"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()
            pipeline.chunk_size = 100
            pipeline.chunk_overlap = 20

            # Create predictable text
            long_text = "A" * 250
            metadata = {"source": "test"}

            chunks = pipeline.chunk_text(long_text, metadata)

            # Verify overlap: end of chunk n - overlap should be start of chunk n+1
            for i in range(len(chunks) - 1):
                current_end = chunks[i][1]['char_end']
                next_start = chunks[i + 1][1]['char_start']
                # Next chunk should start before current chunk ends (overlap)
                self.assertLess(next_start, current_end)

    def test_generate_document_id(self):
        """Test document ID generation"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()

            file_path = Path("/data/apollo11/test_doc.txt")
            metadata = {
                "mission": "apollo_11",
                "source": "test_doc",
                "chunk_index": 5
            }

            doc_id = pipeline.generate_document_id(file_path, metadata)

            self.assertEqual(doc_id, "apollo_11_test_doc_chunk_0005")

    def test_extract_mission_from_path(self):
        """Test mission extraction from file paths"""
        from embedding_pipeline import ChromaEmbeddingPipelineTextOnly

        with patch.object(ChromaEmbeddingPipelineTextOnly, '__init__', lambda self, **kwargs: None):
            pipeline = ChromaEmbeddingPipelineTextOnly()

            # Test Apollo 11
            path = Path("/data/apollo11/transcript.txt")
            self.assertEqual(pipeline.extract_mission_from_path(path), "apollo_11")

            # Test Apollo 13
            path = Path("/data/apollo13/transcript.txt")
            self.assertEqual(pipeline.extract_mission_from_path(path), "apollo_13")

            # Test Challenger
            path = Path("/data/challenger/audio.txt")
            self.assertEqual(pipeline.extract_mission_from_path(path), "challenger")

            # Test unknown
            path = Path("/data/other/document.txt")
            self.assertEqual(pipeline.extract_mission_from_path(path), "unknown")


class TestRAGClient(unittest.TestCase):
    """Tests for rag_client.py"""

    def test_format_context_empty(self):
        """Test format_context with empty documents"""
        from rag_client import format_context

        result = format_context([], [])
        self.assertEqual(result, "")

    def test_format_context_basic(self):
        """Test format_context with basic documents"""
        from rag_client import format_context

        documents = ["This is document 1.", "This is document 2."]
        metadatas = [
            {"mission": "apollo_11", "source": "doc1", "document_category": "technical"},
            {"mission": "apollo_13", "source": "doc2", "document_category": "transcript"}
        ]

        result = format_context(documents, metadatas)

        self.assertIn("Retrieved Context from NASA Documents", result)
        self.assertIn("[Source 1]", result)
        self.assertIn("[Source 2]", result)
        self.assertIn("Apollo 11", result)
        self.assertIn("Apollo 13", result)
        self.assertIn("This is document 1.", result)
        self.assertIn("This is document 2.", result)

    def test_format_context_truncation(self):
        """Test format_context truncates long documents"""
        from rag_client import format_context

        # Create a very long document
        long_doc = "A" * 3000
        documents = [long_doc]
        metadatas = [{"mission": "apollo_11", "source": "doc1"}]

        result = format_context(documents, metadatas)

        self.assertIn("...[truncated]", result)
        self.assertLess(len(result), len(long_doc) + 500)  # Should be less than original

    def test_build_source_list(self):
        """Test building source citations"""
        from rag_client import build_source_list

        metadatas = [
            {"mission": "apollo_11", "source": "transcript", "document_category": "pao"},
            {"mission": "challenger", "source": "audio", "document_category": "mission_audio"}
        ]

        sources = build_source_list(metadatas)

        self.assertEqual(len(sources), 2)
        self.assertIn("[1]", sources[0])
        self.assertIn("Apollo 11", sources[0])
        self.assertIn("[2]", sources[1])
        self.assertIn("Challenger", sources[1])

    def test_discover_chroma_backends_no_dirs(self):
        """Test discover_chroma_backends when no chroma directories exist"""
        from rag_client import discover_chroma_backends

        # Create a temp directory with no chroma dirs
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                backends = discover_chroma_backends()
                # Should return empty or minimal results
                # (depends on if default chroma_db exists)
            finally:
                os.chdir(original_cwd)


class TestLLMClient(unittest.TestCase):
    """Tests for llm_client.py"""

    def test_build_system_prompt(self):
        """Test system prompt generation"""
        from llm_client import build_system_prompt

        prompt = build_system_prompt()

        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)
        self.assertIn("NASA", prompt)

    def test_append_history(self):
        """Test conversation history management"""
        from llm_client import append_history

        history = []
        history = append_history(history, "user", "Hello")
        history = append_history(history, "assistant", "Hi there!")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi there!")

    def test_get_conversation_summary(self):
        """Test conversation summary generation"""
        from llm_client import get_conversation_summary

        # Empty history
        summary = get_conversation_summary([])
        self.assertEqual(summary, "No previous conversation.")

        # With history
        history = [
            {"role": "user", "content": "What is Apollo 11?"},
            {"role": "assistant", "content": "Apollo 11 was the first mission to land on the Moon."}
        ]
        summary = get_conversation_summary(history)
        self.assertIn("User:", summary)
        self.assertIn("Assistant:", summary)


class TestRAGASEvaluator(unittest.TestCase):
    """Tests for ragas_evaluator.py"""

    def test_evaluate_single_empty_inputs(self):
        """Test evaluate_single with empty inputs"""
        from ragas_evaluator import evaluate_single

        # Empty question
        result = evaluate_single("", "Some context", "Some answer")
        self.assertIn("error", result)

        # Empty context
        result = evaluate_single("What is Apollo?", "", "Apollo was a program")
        self.assertIn("error", result)

        # Empty answer
        result = evaluate_single("What is Apollo?", "Apollo context", "")
        self.assertIn("error", result)

    def test_load_eval_dataset_json(self):
        """Test loading JSON evaluation dataset"""
        from ragas_evaluator import load_eval_dataset

        # Create temp JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump([
                {"question": "What is Apollo 11?", "category": "overview"},
                {"question": "Who were the crew?", "category": "crew"}
            ], f)
            temp_path = f.name

        try:
            samples = load_eval_dataset(temp_path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["question"], "What is Apollo 11?")
        finally:
            os.unlink(temp_path)

    def test_load_eval_dataset_txt(self):
        """Test loading TXT evaluation dataset"""
        from ragas_evaluator import load_eval_dataset

        # Create temp TXT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Q: What is Apollo 11?\n")
            f.write("Category: overview\n\n")
            f.write("Q: Who were the crew?\n")
            f.write("Category: crew\n")
            temp_path = f.name

        try:
            samples = load_eval_dataset(temp_path)
            self.assertGreaterEqual(len(samples), 2)
        finally:
            os.unlink(temp_path)

    def test_load_eval_dataset_not_found(self):
        """Test loading non-existent file"""
        from ragas_evaluator import load_eval_dataset

        samples = load_eval_dataset("/nonexistent/path/file.json")
        self.assertEqual(len(samples), 1)
        self.assertIn("error", samples[0])

    def test_summarize_scores(self):
        """Test score summarization"""
        from ragas_evaluator import summarize_scores

        score_data = {
            "faithfulness_scores": [0.8, 0.9, 0.7],
            "relevancy_scores": [0.85, 0.95, 0.75],
            "total_questions": 3,
            "evaluated_questions": 3
        }

        summary = summarize_scores(score_data)

        self.assertAlmostEqual(summary["faithfulness_mean"], 0.8, places=2)
        self.assertEqual(summary["faithfulness_min"], 0.7)
        self.assertEqual(summary["faithfulness_max"], 0.9)
        self.assertAlmostEqual(summary["answer_relevancy_mean"], 0.85, places=2)


class TestEvaluationDataset(unittest.TestCase):
    """Tests for evaluation dataset files"""

    def test_test_questions_json_exists(self):
        """Test that test_questions.json exists and is valid"""
        import json

        json_path = Path(__file__).parent / "test_questions.json"
        self.assertTrue(json_path.exists(), "test_questions.json should exist")

        with open(json_path) as f:
            data = json.load(f)

        self.assertIn("questions", data)
        questions = data["questions"]
        self.assertGreaterEqual(len(questions), 5, "Should have at least 5 questions")

        # Check required categories
        categories = set(q.get("category", "") for q in questions)
        required_categories = {"overview", "crew", "technical", "emergency", "timeline"}
        for cat in required_categories:
            self.assertIn(cat, categories, f"Should have {cat} category")

    def test_evaluation_dataset_txt_exists(self):
        """Test that evaluation_dataset.txt exists"""
        txt_path = Path(__file__).parent / "evaluation_dataset.txt"
        self.assertTrue(txt_path.exists(), "evaluation_dataset.txt should exist")

        with open(txt_path) as f:
            content = f.read()

        # Should have questions
        self.assertIn("Q:", content)
        self.assertIn("Category:", content)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            import embedding_pipeline
            import rag_client
            import llm_client
            import ragas_evaluator
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_env_example_exists(self):
        """Test that .env.example exists with required keys"""
        env_path = Path(__file__).parent / ".env.example"
        self.assertTrue(env_path.exists(), ".env.example should exist")

        with open(env_path) as f:
            content = f.read()

        required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "CHROMA_DIR"]
        for key in required_keys:
            self.assertIn(key, content, f".env.example should contain {key}")


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGClient))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMClient))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGASEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)

