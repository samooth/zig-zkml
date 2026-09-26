# Registro de decisiones (ADR)

Cada decisión de diseño que afecta al soundness, al commitment o al statement
se registra aquí como un ADR numerado. Los ADR son **inmutables**: si una
decisión cambia, se añade un ADR nuevo que la sustituye y se marca el anterior
como reemplazado.

La especificación vigente es [BLUE_PRINT.md](../BLUE_PRINT.md); este directorio
explica **por qué** es como es, no **qué** es.

| ADR | Título | Estado |
|---|---|---|
| [ADR-0001](ADR-0001-zkml-vs-blueprint.md) | Abandonar el diseño de zkML.md en favor del contrato de aritmética exacta | aceptado |

## Plantilla

```markdown
# ADR-NNNN — <título en una línea>

- **Estado:** propuesto | aceptado | reemplazado por ADR-NNNN
- **Fecha:** AAAA-MM
- **Ámbito:** <qué parte del sistema afecta>
- **Vigente en:** <sección de BLUE_PRINT.md>

## Contexto
## Decisión
## Consecuencias
## Alternativas descartadas
## Referencias
```
