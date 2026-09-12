# quant_ui

Streamlit UI for the cedge platform. Every page is a thin client over the
`services/` HTTP APIs

## Run

Start the services this app depends on, then:

    pip install -r apps/quant_ui/requirements.txt
    streamlit run apps/quant_ui/app.py

| Variable | Default | Service |
|---|---|---|
| `CEDGE_PORTFOLIO_PERFORMANCE_API_URL` | `http://localhost:8000` | portfolio_performance |
| `CEDGE_RISK_API_URL` | `http://localhost:8001` | risk |
| `CEDGE_MARKETDATA_API_URL` | `http://localhost:8002` | marketdata |