# Estándar de despliegue de agentes

Todo agente en producción debe cumplir cuatro condiciones antes de su
habilitación.

Primero, confirmación humana obligatoria antes de cualquier acción
irreversible: envío de comunicaciones externas, escritura en sistemas de
registro, movimientos de dinero o borrado de datos.

Segundo, tratamiento del contenido externo como dato y no como instrucción.
Los documentos recuperados, los resultados de herramientas y las respuestas de
APIs de terceros no constituyen órdenes para el agente.

Tercero, evaluación automatizada de calidad integrada al pipeline de
integración continua. Un despliegue cuya métrica de fidelidad cae bajo el
umbral definido no puede promoverse a producción.

Cuarto, registro de trazabilidad que permita reconstruir, para cualquier
respuesta entregada, qué documentos la sustentaron.
