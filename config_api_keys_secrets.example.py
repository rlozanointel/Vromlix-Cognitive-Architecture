# -*- coding: utf-8 -*-

# ==============================================================================
# VROMLIX SOTA ROUTING REGISTRY (Single Source of Truth) - PUBLIC MOCK
# ==============================================================================

MAX_FILE_SIZE_MB = 5

# Registro de enrutamiento basado en roles y cuotas
MODEL_ROUTING_REGISTRY = {
    "PRECISION": {
        "model_id": "gemini-3-flash-preview",
        "role": "Lógica compleja, Código, Auditoría, Multimodal",
        "rpd_per_key": 20,
        "rpm_per_key": 5,
        "tpm_limit": 250000,
    },
    "VOLUMEN": {
        "model_id": "gemini-3.1-flash-lite-preview",
        "role": "Triaje masivo, Extracción JSON, RAG",
        "rpd_per_key": 500,
        "rpm_per_key": 15,
        "tpm_limit": 250000,
    },
    "MASIVO": {
        "model_id": "gemma-3-27b-it",
        "role": "Procesamiento de texto puro, Open-Weights",
        "rpd_per_key": 14400,
        "rpm_per_key": 15,
        "tpm_limit": 250000,
    },
}

# ------------------------------------------------------------------------------
# CREDENCIALES (Variables de entorno inyectadas en runtime)
# ------------------------------------------------------------------------------

# Lista de rotación de llaves para evadir límites HTTP 429
API_KEYS_DB = [
    {
        "proveedor": "gemini",
        "cuenta": "mock.account.1",
        "proyecto": "Colab-Worker-01",
        "key": "Key_01",
        "api_key": "YOUR_API_KEY_HERE",
    },
    {
        "proveedor": "gemini",
        "cuenta": "mock.account.2",
        "proyecto": "Colab-Worker-02",
        "key": "Key_02",
        "api_key": "YOUR_API_KEY_HERE",
    },
]
