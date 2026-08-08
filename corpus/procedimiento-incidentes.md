# Procedimiento de gestión de incidentes de IA

Un incidente de IA se declara cuando un sistema basado en modelos de lenguaje
produce una salida que causa daño, expone información restringida o toma una
acción no autorizada.

Los niveles de severidad son tres. Severidad 1: exposición de datos de cliente
o acción irreversible ejecutada sin autorización; el plazo de respuesta es de
1 hora. Severidad 2: salida incorrecta que llegó a un usuario final; plazo de
4 horas. Severidad 3: comportamiento anómalo detectado en pruebas o monitoreo,
sin impacto externo; plazo de 48 horas.

Todo incidente de severidad 1 requiere desactivar el agente afectado antes de
iniciar el diagnóstico. La reactivación exige aprobación del responsable de
gobernanza de IA y evidencia de que el control que falló fue corregido.
