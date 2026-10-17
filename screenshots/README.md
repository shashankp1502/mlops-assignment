# Screenshots checklist

CLI-based verification output is already captured here as text
(`helm_deployment_verification.txt`, `monitoring_verification.txt`) from
an actual end-to-end Minikube + Helm deployment.

For the final written report, add real GUI screenshots (PNG):

- [ ] `docker images` / `docker ps` showing the built `heart-risk-api` image and a running (healthy) container
- [ ] GitHub Actions run — all 3 jobs (quality, train, containerize) green
- [ ] GitHub Actions `train` job log showing both models' metrics
- [ ] `kubectl -n heart-risk get pods,svc,ingress` (2/2 API replicas + Prometheus + Grafana Running)
- [ ] `helm list -n heart-risk` showing the deployed release
- [ ] Browser hitting `/docs` (FastAPI Swagger UI) and a live `/predict` call
- [ ] Prometheus UI `/targets` showing `heart-risk-api` as **UP**
- [ ] Grafana dashboard "Heart Risk API — Service Health" with live data (pie chart, gauge, latency panels)
- [ ] MLflow UI comparing the Decision Tree vs Gradient Boosting runs, plus the registered model version in the Model Registry tab
- [ ] MLflow run artifacts tab showing the feature-importance plot

Naming convention: `0X_description.png` (e.g. `01_docker_build.png`,
`02_github_actions_pipeline.png`, `03_k8s_helm_pods.png`,
`04_grafana_dashboard.png`).
