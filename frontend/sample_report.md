# RAGAS Ablation Report

_Sample / placeholder data for the demo. Replace by running the eval harness and
committing a real report to `backend/data/eval/reports/`._

## Metrics by stage

| Stage         | Faithfulness | Context Precision | Context Recall | Answer Relevancy | Delta  |
| ------------- | ------------ | ----------------- | -------------- | ---------------- | ------ |
| naive         | 0.71         | 0.62              | 0.58           | 0.74             | —      |
| hybrid        | 0.79         | 0.73              | 0.69           | 0.80             | +0.08  |
| +graph        | 0.85         | 0.80              | 0.78           | 0.84             | +0.06  |
| +multimodal   | 0.88         | 0.83              | 0.81           | 0.86             | +0.03  |

Delta is the change in Faithfulness relative to the previous stage.

## Failures

| Question                                      | Stage  | Failure mode                                  |
| --------------------------------------------- | ------ | --------------------------------------------- |
| "What is the current refund SOP version?"     | naive  | Cited a superseded SOP version.               |
| "List the steps in the onboarding checklist." | hybrid | Missed a step that lived only in a table.     |
