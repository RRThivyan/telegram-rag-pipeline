# MLOps — Frequently Asked Questions

## What is MLOps?
MLOps (Machine Learning Operations) is a set of practices that combines ML, DevOps, and Data Engineering to reliably deploy and maintain ML models in production. The goal is to shorten the cycle from experimentation to production and ensure models remain accurate over time.

## What is model drift?
Model drift occurs when a deployed model's performance degrades over time because the statistical properties of the real-world data change. There are two types:
- **Data drift (covariate shift)**: the distribution of input features changes.
- **Concept drift**: the relationship between inputs and outputs changes (e.g., user behaviour shifts after a market event).

## What is a feature store?
A feature store is a centralised repository for storing, versioning, and serving ML features. It ensures training and serving use the same feature logic, preventing training-serving skew. Popular feature stores include Feast, Tecton, and Vertex AI Feature Store.

## What is CI/CD for ML?
Continuous Integration and Continuous Deployment (CI/CD) for ML automates the pipeline from code commit to production deployment. In ML, this includes: running unit tests, data validation, model training, evaluation against a baseline, and automated deployment if metrics pass a threshold. Tools: MLflow, Kubeflow, GitHub Actions, and DVC.

## What is model versioning?
Model versioning tracks different versions of a trained model — including the code, data, hyperparameters, and evaluation metrics that produced it. This enables reproducibility and rollback. MLflow, DVC, and Weights & Biases are popular tools.

## What is A/B testing in ML?
A/B testing routes a fraction of production traffic to a new model (B) while the existing model (A) handles the rest. By comparing business metrics (click-through rate, conversion, etc.) between groups, teams can validate whether the new model delivers real-world improvement before full rollout.

## What is shadow deployment?
In shadow deployment, the new model runs in parallel with the existing model, processing real requests but not serving its responses to users. This allows teams to compare outputs and latency without any user-facing risk, before promoting the new model to production.

## What is model monitoring?
Model monitoring tracks a deployed model's behaviour in production. Key metrics include: prediction distribution, input feature statistics, latency, error rate, and downstream business KPIs. Alerts fire when metrics deviate beyond thresholds, triggering retraining or rollback.
