"""
DeepGuard v2 - Gemini Explainability Layer

Uses Gemini API to generate human-readable explanations.
Key principles:
- Gemini explains, it doesn't detect
- Deterministic fallback always available
- Caching to avoid API spam
- Rate limiting handled gracefully
"""

import time
from typing import Optional, Dict
from dataclasses import dataclass

from config import config
from core.confidence import ConfidenceLevel, DetectionResult


@dataclass
class Explanation:
    """Explanation result"""
    text: str
    source: str  # "gemini" or "fallback"
    cached: bool = False


class GeminiExplainer:
    """
    Gemini-powered explanation generator.
    
    Converts technical detection signals into human-understandable text.
    Falls back to deterministic templates when API is unavailable.
    """
    
    def __init__(self):
        self.cfg = config.gemini
        self.client = None
        self._init_client()
        
        # Explanation cache
        self.cache: Dict[str, Explanation] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
    def _init_client(self):
        """Initialize Gemini client if API key available"""
        if not self.cfg.api_key or not self.cfg.enabled:
            return
            
        try:
            from google import genai
            self.client = genai.Client(api_key=self.cfg.api_key)
        except Exception as e:
            print(f"[WARN] Gemini client init failed: {e}")
            self.client = None
    
    def explain(
        self,
        result: DetectionResult,
        context: str = "video",
        trend: str = "stable",
        frames_analyzed: int = 0
    ) -> Explanation:
        """
        Generate explanation for a detection result.
        
        Args:
            result: Detection result to explain
            context: Type of content (video, image, stream)
            trend: Score trend (rising, falling, stable)
            frames_analyzed: Number of frames in temporal window
            
        Returns:
            Explanation with text and source
        """
        
        # Check cache first
        cache_key = f"{result.level.value}_{result.confidence_pct}_{trend}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Try Gemini if available
        if self.client and self.cfg.enabled:
            explanation = self._explain_with_gemini(result, context, trend, frames_analyzed)
            if explanation:
                self._cache(cache_key, explanation)
                return explanation
        
        # Fallback to deterministic explanation
        explanation = self._fallback_explain(result, context, trend, frames_analyzed)
        self._cache(cache_key, explanation)
        return explanation
    
    def _explain_with_gemini(
        self,
        result: DetectionResult,
        context: str,
        trend: str,
        frames_analyzed: int
    ) -> Optional[Explanation]:
        """Generate explanation using Gemini API"""
        
        try:
            prompt = self._build_prompt(result, context, trend, frames_analyzed)
            
            response = self.client.models.generate_content(
                model=self.cfg.model,
                contents=prompt
            )
            
            if response and response.text:
                return Explanation(
                    text=response.text.strip(),
                    source="gemini"
                )
                
        except Exception as e:
            print(f"[WARN] Gemini API error: {e}")
            
        return None
    
    def _build_prompt(
        self,
        result: DetectionResult,
        context: str,
        trend: str,
        frames_analyzed: int
    ) -> str:
        """Build prompt for Gemini"""
        
        return f"""You are an AI safety assistant explaining deepfake detection results.

Detection Result:
- Classification: {result.level.value}
- Confidence Score: {result.score:.2%} probability of being manipulated
- Confidence Display: {result.confidence_pct}%
- Content Type: {context}
- Prediction Trend: {trend}
- Frames Analyzed: {frames_analyzed}

Generate a brief, helpful explanation (2-3 sentences) for a non-technical user.
- If REAL/LIKELY REAL: Reassure them, mention what looks authentic
- If UNCERTAIN: Explain why it's unclear, suggest caution
- If LIKELY FAKE/DEEPFAKE: Explain what signals triggered this, but avoid alarmism

Be concise. Do not use technical jargon. Do not use markdown formatting."""

    def _fallback_explain(
        self,
        result: DetectionResult,
        context: str,
        trend: str,
        frames_analyzed: int
    ) -> Explanation:
        """Generate deterministic fallback explanation"""
        
        pct = result.confidence_pct
        
        explanations = {
            ConfidenceLevel.REAL: [
                f"This {context} appears to be authentic. Facial features and movements are consistent with natural human expression.",
                f"No manipulation detected. The {context} shows natural facial characteristics across {frames_analyzed} analyzed frames.",
            ],
            ConfidenceLevel.LIKELY_REAL: [
                f"This {context} is most likely authentic. Minor variations detected are within normal range.",
                f"Appears genuine. Slight anomalies may be due to compression or lighting, not manipulation.",
            ],
            ConfidenceLevel.UNCERTAIN: [
                f"Unable to determine with confidence. This could be due to video quality, lighting, or compression artifacts.",
                f"Analysis inconclusive. Consider the source credibility before making judgments.",
                f"The {context} shows mixed signals. Exercise caution and verify through other means.",
            ],
            ConfidenceLevel.LIKELY_FAKE: [
                f"Potential manipulation detected with {pct}% confidence. Facial texture inconsistencies observed.",
                f"This {context} shows signs that may indicate synthetic generation. Verify the source.",
            ],
            ConfidenceLevel.DEEPFAKE: [
                f"High probability of deepfake detected ({pct}%). Facial artifacts and unnatural patterns identified across frames.",
                f"This {context} shows strong indicators of AI manipulation. Facial boundaries and textures appear synthetic.",
                f"Warning: Likely deepfake content. Detected inconsistent facial motion and texture artifacts.",
            ],
        }
        
        # Add trend-specific context
        trend_suffix = ""
        if trend == "rising":
            trend_suffix = " Detection confidence is increasing."
        elif trend == "falling":
            trend_suffix = " Detection confidence is decreasing."
        
        import random
        base_text = random.choice(explanations.get(result.level, ["Analysis complete."]))
        
        return Explanation(
            text=base_text + trend_suffix,
            source="fallback"
        )
    
    def _get_cached(self, key: str) -> Optional[Explanation]:
        """Get cached explanation if still valid"""
        if key not in self.cache:
            return None
            
        timestamp = self.cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.cfg.cache_duration:
            del self.cache[key]
            del self.cache_timestamps[key]
            return None
            
        explanation = self.cache[key]
        explanation.cached = True
        return explanation
    
    def _cache(self, key: str, explanation: Explanation):
        """Cache an explanation"""
        self.cache[key] = explanation
        self.cache_timestamps[key] = time.time()
    
    def clear_cache(self):
        """Clear explanation cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
