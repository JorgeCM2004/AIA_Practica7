from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def anonymize_node(user_msg):
	analyzer = AnalyzerEngine(supported_languages=["en"])
	anonymizer = AnonymizerEngine()
	results = analyzer.analyze(text=user_msg, entities=[], language="en")

	anonymized_result = anonymizer.anonymize(text=user_msg, analyzer_results=results)
	safe_text = anonymized_result.text

	return safe_text
