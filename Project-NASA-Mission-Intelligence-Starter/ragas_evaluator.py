import os
import json
from typing import Dict, List, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAGAS imports
RAGAS_AVAILABLE = False
RAGAS_IMPORT_ERROR = ""

try:
    from ragas import SingleTurnSample, evaluate
    from ragas.metrics import ResponseRelevancy, Faithfulness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_IMPORT_ERROR = str(e)


def get_evaluator_llm():
    """Create evaluator LLM with langchain wrapper"""
    if not RAGAS_AVAILABLE:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0
    )
    return LangchainLLMWrapper(llm)


def get_evaluator_embeddings():
    """Create evaluator embeddings with langchain wrapper"""
    if not RAGAS_AVAILABLE:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key
    )
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, Union[float, str]]:
    """Evaluate response quality using RAGAS metrics

    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context strings

    Returns:
        Dictionary containing evaluation scores
    """
    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS not available: {RAGAS_IMPORT_ERROR}"}

    # Validate inputs
    if not question or not question.strip():
        return {"error": "Question cannot be empty", "status": "error"}

    if not answer or not answer.strip():
        return {"error": "Answer cannot be empty", "status": "error"}

    if not contexts or all(not c.strip() for c in contexts if c):
        return {"error": "Contexts cannot be empty", "status": "error"}

    try:
        # Create evaluator LLM with model gpt-4o-mini
        evaluator_llm = get_evaluator_llm()

        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = get_evaluator_embeddings()

        if not evaluator_llm or not evaluator_embeddings:
            return {"error": "Failed to initialize RAGAS evaluator components", "status": "error"}

        # Define an instance for each metric to evaluate
        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

        # Clean and prepare contexts
        clean_contexts = [c.strip() for c in contexts if c and c.strip()]

        # Create sample for evaluation
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=clean_contexts
        )

        # Evaluate the response using the metrics
        results = {}

        # Evaluate faithfulness
        try:
            faithfulness_score = faithfulness_metric.single_turn_score(sample)
            results['faithfulness'] = float(faithfulness_score) if faithfulness_score is not None else 0.0
        except Exception as e:
            results['faithfulness'] = 0.0
            results['faithfulness_error'] = str(e)

        # Evaluate response relevancy
        try:
            relevancy_score = relevancy_metric.single_turn_score(sample)
            results['answer_relevancy'] = float(relevancy_score) if relevancy_score is not None else 0.0
        except Exception as e:
            results['answer_relevancy'] = 0.0
            results['answer_relevancy_error'] = str(e)

        results['status'] = 'ok'

        # Return the evaluation results
        return results

    except Exception as e:
        return {
            "error": f"Evaluation failed: {str(e)}",
            "status": "error"
        }


def evaluate_single(question: str, context: Union[str, List[str]], answer: str) -> Dict[str, Any]:
    """Evaluate a single (question, context, answer) triple

    Args:
        question: The user's question
        context: The retrieved context string or list of strings
        answer: The generated answer

    Returns:
        Dictionary containing evaluation scores
    """
    # Convert context string to list if necessary
    if isinstance(context, str):
        contexts = [context]
    else:
        contexts = list(context)

    return evaluate_response_quality(question, answer, contexts)


def load_eval_dataset(path: str) -> List[Dict]:
    """Load evaluation dataset from file

    Args:
        path: Path to the evaluation dataset file

    Returns:
        List of evaluation samples
    """
    samples = []

    try:
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and 'questions' in data:
                    samples = data['questions']
                else:
                    samples = [data]
        elif path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse simple text format
                lines = content.strip().split('\n')
                current_sample = {}
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        if current_sample:
                            samples.append(current_sample)
                            current_sample = {}
                        continue
                    if line.startswith('Q:') or line.startswith('Question:'):
                        if current_sample:
                            samples.append(current_sample)
                        current_sample = {'question': line.split(':', 1)[1].strip()}
                    elif line.startswith('Category:'):
                        current_sample['category'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Expected:') or line.startswith('Ground Truth:'):
                        current_sample['ground_truth'] = line.split(':', 1)[1].strip()
                if current_sample:
                    samples.append(current_sample)
        else:
            return [{"error": f"Unsupported file format: {path}"}]

    except FileNotFoundError:
        return [{"error": f"File not found: {path}"}]
    except Exception as e:
        return [{"error": f"Error loading dataset: {str(e)}"}]

    return samples


def evaluate_batch(dataset_path: str, rag_retrieve_fn=None, llm_generate_fn=None) -> Dict[str, Any]:
    """Evaluate a batch of questions from a dataset file

    Args:
        dataset_path: Path to the evaluation dataset
        rag_retrieve_fn: Optional function to retrieve context (signature: fn(question) -> List[str])
        llm_generate_fn: Optional function to generate answer (signature: fn(question, context) -> str)

    Returns:
        Dictionary containing per-question scores and aggregate summary
    """
    # Load the dataset
    samples = load_eval_dataset(dataset_path)

    if not samples:
        return {"error": "No samples found in dataset", "status": "error"}

    if samples and 'error' in samples[0]:
        return {"error": samples[0]['error'], "status": "error"}

    results = {
        "per_question": [],
        "summary": {},
        "status": "ok"
    }

    faithfulness_scores = []
    relevancy_scores = []

    for i, sample in enumerate(samples):
        question = sample.get('question', '')
        if not question:
            continue

        # Get context - either from sample or from retrieval function
        context = sample.get('context', sample.get('contexts', []))
        if not context and rag_retrieve_fn:
            try:
                context = rag_retrieve_fn(question)
            except Exception as e:
                context = [f"Error retrieving context: {str(e)}"]

        if isinstance(context, str):
            context = [context]

        # Get answer - either from sample or from generation function
        answer = sample.get('answer', sample.get('response', ''))
        if not answer and llm_generate_fn:
            try:
                answer = llm_generate_fn(question, "\n".join(context))
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"

        # Skip if we don't have all necessary parts
        if not answer or not context:
            results["per_question"].append({
                "question": question,
                "error": "Missing answer or context",
                "category": sample.get('category', 'unknown')
            })
            continue

        # Evaluate
        scores = evaluate_response_quality(question, answer, context)

        result_entry = {
            "question": question,
            "category": sample.get('category', 'unknown'),
            "scores": scores
        }

        results["per_question"].append(result_entry)

        # Collect scores for summary
        if 'faithfulness' in scores and isinstance(scores['faithfulness'], (int, float)):
            faithfulness_scores.append(scores['faithfulness'])
        if 'answer_relevancy' in scores and isinstance(scores['answer_relevancy'], (int, float)):
            relevancy_scores.append(scores['answer_relevancy'])

    # Calculate summary statistics
    results["summary"] = summarize_scores({
        "faithfulness_scores": faithfulness_scores,
        "relevancy_scores": relevancy_scores,
        "total_questions": len(samples),
        "evaluated_questions": len(faithfulness_scores)
    })

    return results


def summarize_scores(score_data: Dict) -> Dict[str, Any]:
    """Summarize evaluation scores

    Args:
        score_data: Dictionary containing score lists

    Returns:
        Summary statistics dictionary
    """
    summary = {}

    faithfulness_scores = score_data.get('faithfulness_scores', [])
    relevancy_scores = score_data.get('relevancy_scores', [])

    if faithfulness_scores:
        summary['faithfulness_mean'] = sum(faithfulness_scores) / len(faithfulness_scores)
        summary['faithfulness_min'] = min(faithfulness_scores)
        summary['faithfulness_max'] = max(faithfulness_scores)

    if relevancy_scores:
        summary['answer_relevancy_mean'] = sum(relevancy_scores) / len(relevancy_scores)
        summary['answer_relevancy_min'] = min(relevancy_scores)
        summary['answer_relevancy_max'] = max(relevancy_scores)

    summary['total_questions'] = score_data.get('total_questions', 0)
    summary['evaluated_questions'] = score_data.get('evaluated_questions', 0)

    return summary
