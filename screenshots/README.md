# Screenshots index

Real screenshots captured from an actual end-to-end run of this project
(local Docker, Minikube + Helm deployment, and MLflow). Embedded directly
in `docs/report.md` / `docs/report.docx`.

| File | Shows |
|---|---|
| `heart_risk_swagger_api.png` | FastAPI Swagger UI (`/docs`) — auto-generated API documentation |
| `k8s-pods.jpg` | `kubectl get pods -n heart-risk` — 2/2 API replicas + Prometheus + Grafana all Running |
| `prometheus.png` | Prometheus `/targets` page — `heart-risk-api` scrape target UP |
| `grafana.png` | Grafana "Heart Risk API — Service Health" dashboard with live traffic (pie chart, gauge, dual-line latency) |
| `model_comaprision.png` | MLflow UI — Decision Tree vs Gradient Boosting run comparison (params/tags) |
| `model/comparision_matrix.png` | MLflow UI — same comparison, metrics section |
| `model/decision_tree_confusion_matrix.png` | MLflow artifact |
| `model/decision_tree_roc_curve.png` | MLflow artifact |
| `model/decision_tree_feature_importance.png` | MLflow artifact |
| `model/gradient_boosting_confusion_matrix.png` | MLflow artifact |
| `model/gradient_boosting_roc_curve.png` | MLflow artifact |
| `model/gradient_boosting_feature_importance.png` | MLflow artifact |
