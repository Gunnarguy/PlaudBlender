"""
LLM-as-a-Judge RAG Evaluation Module for PlaudBlender

Implements automated quality evaluation for RAG outputs using Gemini:
- Faithfulness: Is the answer supported by the retrieved context?
- Answer Relevance: Does it actually address the user's question?
- Context Precision: Did we retrieve the right documents?
- Context Recall: Did we retrieve all relevant documents?
- Hallucination Detection: Are there unsupported claims?

This enables:
1. Automated quality monitoring in production
2. A/B testing of retrieval strategies
3. Regression detection when changing models/settings
4. User feedback calibration against "Gold Set"

Reference: gemini-deep-research2.txt Section 7.1 "LLM-as-a-Judge"
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EvaluationMetric(str, Enum):
    """RAG evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    HALLUCINATION = "hallucination"


@dataclass
class MetricScore:
    """Score for a single evaluation metric."""
    metric: EvaluationMetric
    score: float  # 0.0 - 1.0
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "metric": self.metric.value,
            "score": round(self.score, 3),
            "reasoning": self.reasoning,
            "evidence": self.evidence,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation of a RAG response."""
    query: str
    answer: str
    context_chunks: List[str]
    
    # Individual metric scores
    faithfulness: Optional[MetricScore] = None
    answer_relevance: Optional[MetricScore] = None
    context_precision: Optional[MetricScore] = None
    hallucination: Optional[MetricScore] = None
    
    # Aggregate
    overall_score: float = 0.0
    pass_threshold: bool = False
    evaluation_latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def metrics_dict(self) -> Dict[str, float]:
        """Get all metric scores as dict."""
        metrics = {}
        if self.faithfulness:
            metrics["faithfulness"] = self.faithfulness.score
        if self.answer_relevance:
            metrics["answer_relevance"] = self.answer_relevance.score
        if self.context_precision:
            metrics["context_precision"] = self.context_precision.score
        if self.hallucination:
            metrics["hallucination"] = self.hallucination.score
        return metrics
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query[:100],
            "answer_preview": self.answer[:200],
            "context_count": len(self.context_chunks),
            "metrics": self.metrics_dict,
            "overall_score": round(self.overall_score, 3),
            "pass_threshold": self.pass_threshold,
            "evaluation_latency_ms": self.evaluation_latency_ms,
            "timestamp": self.timestamp,
        }


class RAGEvaluator:
    """
    LLM-as-a-Judge evaluator for RAG quality assessment.
    
    Uses Gemini to score RAG outputs on multiple dimensions:
    - Faithfulness: Answer is grounded in provided context
    - Answer Relevance: Answer addresses the actual question
    - Context Precision: Retrieved chunks are relevant
    - Hallucination: No made-up facts
    """
    
    # Default thresholds
    FAITHFULNESS_THRESHOLD = float(os.getenv("RAG_FAITHFULNESS_THRESHOLD", "0.8"))
    RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.7"))
    OVERALL_THRESHOLD = float(os.getenv("RAG_OVERALL_THRESHOLD", "0.75"))
    
    # Prompt templates
    FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI answer is fully supported by the provided context.

Question: {query}

Context Documents:
{context}

AI Answer: {answer}

Evaluate FAITHFULNESS: Is every claim in the answer directly supported by the context?

Score from 0.0 to 1.0:
- 1.0: Every statement is explicitly supported by context
- 0.7-0.9: Most statements supported, minor extrapolations
- 0.4-0.6: Some unsupported claims mixed with supported ones
- 0.0-0.3: Mostly unsupported or contradicts context

Respond in JSON format:
{{"score": 0.X, "reasoning": "...", "unsupported_claims": ["claim1", "claim2"]}}"""

    RELEVANCE_PROMPT = """You are an expert evaluator assessing whether an AI answer addresses the user's question.

Question: {query}

AI Answer: {answer}

Evaluate ANSWER RELEVANCE: Does the answer actually address what was asked?

Score from 0.0 to 1.0:
- 1.0: Directly and completely answers the question
- 0.7-0.9: Answers the main question, may miss nuances
- 0.4-0.6: Partially addresses the question, off-topic elements
- 0.0-0.3: Doesn't answer the question or completely off-topic

Respond in JSON format:
{{"score": 0.X, "reasoning": "...", "missing_aspects": ["aspect1", "aspect2"]}}"""

    CONTEXT_PRECISION_PROMPT = """You are an expert evaluator assessing retrieval quality.

Question: {query}

Retrieved Context Documents:
{context}

Evaluate CONTEXT PRECISION: Are the retrieved documents relevant to answering this question?

Score from 0.0 to 1.0:
- 1.0: All documents are highly relevant
- 0.7-0.9: Most documents relevant, few noise
- 0.4-0.6: Mixed relevance, significant noise
- 0.0-0.3: Mostly irrelevant documents

Respond in JSON format:
{{"score": 0.X, "reasoning": "...", "relevant_docs": [0, 1, 2], "irrelevant_docs": [3, 4]}}"""

    HALLUCINATION_PROMPT = """You are an expert evaluator detecting hallucinations in AI answers.

Question: {query}

Context (the ONLY information available):
{context}

AI Answer: {answer}

Evaluate HALLUCINATION: Does the answer contain made-up facts not in the context?

Score from 0.0 to 1.0 (INVERSE - higher is better, meaning LESS hallucination):
- 1.0: No hallucinations detected
- 0.7-0.9: Very minor embellishments
- 0.4-0.6: Some fabricated details
- 0.0-0.3: Significant hallucinations

Respond in JSON format:
{{"score": 0.X, "reasoning": "...", "hallucinated_claims": ["claim1", "claim2"]}}"""

    def __init__(self, model: str = "gemini-1.5-flash"):
        """
        Initialize evaluator.
        
        Args:
            model: Gemini model to use for evaluation
        """
        self.model_name = model
        self._model = None
    
    def _get_model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY required for evaluation")
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
        
        return self._model
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
        metrics: Optional[List[EvaluationMetric]] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG output quality.
        
        Args:
            query: Original user query
            answer: Generated answer
            context_chunks: Retrieved context documents
            metrics: Which metrics to evaluate (default: all)
            
        Returns:
            EvaluationResult with scores and reasoning
        """
        import time
        start = time.time()
        
        if metrics is None:
            metrics = [
                EvaluationMetric.FAITHFULNESS,
                EvaluationMetric.ANSWER_RELEVANCE,
                EvaluationMetric.CONTEXT_PRECISION,
                EvaluationMetric.HALLUCINATION,
            ]
        
        result = EvaluationResult(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
        )
        
        context_str = self._format_context(context_chunks)
        
        # Evaluate each metric
        scores = []
        
        if EvaluationMetric.FAITHFULNESS in metrics:
            result.faithfulness = self._evaluate_metric(
                self.FAITHFULNESS_PROMPT,
                EvaluationMetric.FAITHFULNESS,
                query=query,
                answer=answer,
                context=context_str,
            )
            if result.faithfulness:
                scores.append(result.faithfulness.score)
        
        if EvaluationMetric.ANSWER_RELEVANCE in metrics:
            result.answer_relevance = self._evaluate_metric(
                self.RELEVANCE_PROMPT,
                EvaluationMetric.ANSWER_RELEVANCE,
                query=query,
                answer=answer,
                context=context_str,
            )
            if result.answer_relevance:
                scores.append(result.answer_relevance.score)
        
        if EvaluationMetric.CONTEXT_PRECISION in metrics:
            result.context_precision = self._evaluate_metric(
                self.CONTEXT_PRECISION_PROMPT,
                EvaluationMetric.CONTEXT_PRECISION,
                query=query,
                answer=answer,
                context=context_str,
            )
            if result.context_precision:
                scores.append(result.context_precision.score)
        
        if EvaluationMetric.HALLUCINATION in metrics:
            result.hallucination = self._evaluate_metric(
                self.HALLUCINATION_PROMPT,
                EvaluationMetric.HALLUCINATION,
                query=query,
                answer=answer,
                context=context_str,
            )
            if result.hallucination:
                scores.append(result.hallucination.score)
        
        # Compute overall score
        if scores:
            result.overall_score = sum(scores) / len(scores)
            result.pass_threshold = result.overall_score >= self.OVERALL_THRESHOLD
        
        result.evaluation_latency_ms = (time.time() - start) * 1000
        
        logger.info(
            f"📊 RAG Eval: overall={result.overall_score:.2f} "
            f"{'✅ PASS' if result.pass_threshold else '❌ FAIL'} "
            f"({result.evaluation_latency_ms:.0f}ms)"
        )
        
        return result
    
    def _format_context(self, chunks: List[str], max_chars: int = 8000) -> str:
        """Format context chunks for prompt."""
        formatted = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"[Document {i+1}]: {chunk}"
            if total_chars + len(chunk_text) > max_chars:
                break
            formatted.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n\n".join(formatted)
    
    def _evaluate_metric(
        self,
        prompt_template: str,
        metric: EvaluationMetric,
        **kwargs,
    ) -> Optional[MetricScore]:
        """Evaluate a single metric using LLM."""
        try:
            import google.generativeai as genai
            
            model = self._get_model()
            prompt = prompt_template.format(**kwargs)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                )
            )
            
            # Parse JSON response
            text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            # Extract evidence based on metric
            evidence = []
            if "unsupported_claims" in data:
                evidence = data["unsupported_claims"]
            elif "missing_aspects" in data:
                evidence = data["missing_aspects"]
            elif "hallucinated_claims" in data:
                evidence = data["hallucinated_claims"]
            elif "irrelevant_docs" in data:
                evidence = [f"Doc {i}" for i in data["irrelevant_docs"]]
            
            return MetricScore(
                metric=metric,
                score=float(data.get("score", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                evidence=evidence,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse eval response for {metric.value}: {e}")
            return MetricScore(
                metric=metric,
                score=0.5,
                reasoning=f"Parse error: {e}",
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {metric.value}: {e}")
            return None
    
    def quick_faithfulness_check(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> Tuple[float, str]:
        """
        Quick faithfulness check without full evaluation.
        
        Returns:
            (score, reasoning) tuple
        """
        result = self.evaluate(
            query="[Quick check]",
            answer=answer,
            context_chunks=context_chunks,
            metrics=[EvaluationMetric.FAITHFULNESS],
        )
        
        if result.faithfulness:
            return (result.faithfulness.score, result.faithfulness.reasoning)
        return (0.5, "Evaluation failed")


@dataclass
class GoldSetExample:
    """A gold-standard example for evaluator calibration."""
    query: str
    context: List[str]
    good_answer: str
    bad_answer: str
    expected_good_score: float  # Expected score for good answer
    expected_bad_score: float   # Expected score for bad answer


class EvaluatorCalibrator:
    """
    Calibrate the LLM-as-a-Judge against human-verified examples.
    
    Process:
    1. Create "Gold Set" of examples with known good/bad answers
    2. Run evaluator on Gold Set
    3. Compare evaluator scores to expected scores
    4. Compute alignment score (how well evaluator matches human judgment)
    """
    
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.gold_set: List[GoldSetExample] = []
    
    def add_gold_example(
        self,
        query: str,
        context: List[str],
        good_answer: str,
        bad_answer: str,
        expected_good: float = 0.9,
        expected_bad: float = 0.3,
    ):
        """Add example to gold set."""
        self.gold_set.append(GoldSetExample(
            query=query,
            context=context,
            good_answer=good_answer,
            bad_answer=bad_answer,
            expected_good_score=expected_good,
            expected_bad_score=expected_bad,
        ))
    
    def calibrate(self) -> Dict:
        """
        Run calibration against gold set.
        
        Returns:
            Dict with alignment scores and per-example results
        """
        if not self.gold_set:
            return {"error": "No gold set examples"}
        
        results = []
        good_diffs = []
        bad_diffs = []
        
        for example in self.gold_set:
            # Evaluate good answer
            good_result = self.evaluator.evaluate(
                query=example.query,
                answer=example.good_answer,
                context_chunks=example.context,
            )
            
            # Evaluate bad answer
            bad_result = self.evaluator.evaluate(
                query=example.query,
                answer=example.bad_answer,
                context_chunks=example.context,
            )
            
            good_diff = abs(good_result.overall_score - example.expected_good_score)
            bad_diff = abs(bad_result.overall_score - example.expected_bad_score)
            
            good_diffs.append(good_diff)
            bad_diffs.append(bad_diff)
            
            results.append({
                "query": example.query[:50],
                "good_expected": example.expected_good_score,
                "good_actual": good_result.overall_score,
                "good_diff": good_diff,
                "bad_expected": example.expected_bad_score,
                "bad_actual": bad_result.overall_score,
                "bad_diff": bad_diff,
            })
        
        # Compute alignment (1 - average absolute difference)
        avg_good_diff = sum(good_diffs) / len(good_diffs)
        avg_bad_diff = sum(bad_diffs) / len(bad_diffs)
        alignment = 1 - (avg_good_diff + avg_bad_diff) / 2
        
        return {
            "alignment_score": round(alignment, 3),
            "avg_good_diff": round(avg_good_diff, 3),
            "avg_bad_diff": round(avg_bad_diff, 3),
            "examples_evaluated": len(self.gold_set),
            "results": results,
        }


def get_evaluator(model: str = "gemini-1.5-flash") -> RAGEvaluator:
    """Factory function for RAG evaluator."""
    return RAGEvaluator(model=model)


def evaluate_rag_response(
    query: str,
    answer: str,
    context_chunks: List[str],
) -> EvaluationResult:
    """Quick-access evaluation function."""
    evaluator = get_evaluator()
    return evaluator.evaluate(query, answer, context_chunks)


if __name__ == "__main__":
    # Demo the evaluator
    print("\n" + "="*70)
    print("RAG Evaluation Demo")
    print("="*70)
    
    # Example evaluation
    query = "What were the main action items from the Q4 planning meeting?"
    
    context = [
        "Meeting Notes - Q4 Planning (2025-01-10): Attendees discussed budget allocation. Sarah committed to submitting the revised budget by Friday. Mark will coordinate with engineering on the API timeline.",
        "Follow-up: John mentioned concerns about the tight deadline for the mobile app launch.",
    ]
    
    # Good answer (faithful to context)
    good_answer = "The main action items from the Q4 planning meeting were: 1) Sarah will submit the revised budget by Friday, and 2) Mark will coordinate with engineering on the API timeline."
    
    # Bad answer (includes hallucination)
    bad_answer = "The main action items were: 1) Sarah will submit the revised budget by Friday, 2) Mark will coordinate with engineering, and 3) The team decided to postpone the product launch to Q1 2026."
    
    try:
        evaluator = get_evaluator()
        
        print("\n📊 Evaluating GOOD answer...")
        good_result = evaluator.evaluate(query, good_answer, context)
        print(f"   Overall: {good_result.overall_score:.2f} {'✅ PASS' if good_result.pass_threshold else '❌ FAIL'}")
        print(f"   Metrics: {good_result.metrics_dict}")
        
        print("\n📊 Evaluating BAD answer...")
        bad_result = evaluator.evaluate(query, bad_answer, context)
        print(f"   Overall: {bad_result.overall_score:.2f} {'✅ PASS' if bad_result.pass_threshold else '❌ FAIL'}")
        print(f"   Metrics: {bad_result.metrics_dict}")
        if bad_result.hallucination:
            print(f"   Hallucinations: {bad_result.hallucination.evidence}")
    
    except ValueError as e:
        print(f"\n⚠️  Evaluation requires GOOGLE_API_KEY: {e}")
