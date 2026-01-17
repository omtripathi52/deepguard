from core.gemini_explainer import GeminiExplainer

explainer = GeminiExplainer()

text = explainer.explain(
    label="deepfake",
    score=0.84,
    context="video"
)

print("\n[GEMINI EXPLANATION]\n")
print(text)
