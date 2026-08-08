# Política de acceso a datos

Toda consulta a los repositorios documentales de la firma se ejecuta bajo el
principio de mínimo privilegio. Un principal recibe permiso de lectura sobre
las colecciones asociadas a su unidad de negocio, y nunca permiso de escritura
por defecto.

Las credenciales de acceso a sistemas internos se gestionan exclusivamente por
variables de entorno y bóveda de secretos. Está prohibido incrustar claves,
tokens o contraseñas en el código fuente o en los documentos indexados.

El plazo de retención de registros de auditoría de consultas es de 24 meses.
Cada consulta registra el principal, la colección, los documentos recuperados
y si alguno fue puesto en cuarentena.
