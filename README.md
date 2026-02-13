# DRL Micro1

Este repositorio contiene dos notebooks y codigo compartido para el reto de Deep Reinforcement Learning:

- LunarLander con DQN (SB3)
- Assault (Atari) con DQN + Prioritized Experience Replay (PER) en PyTorch

## Ejecucion en Colab

1. Abre el notebook desde la carpeta `notebooks/`.
2. Ejecuta la celda de instalacion.
3. (Opcional) Monta Google Drive para persistir artefactos.
4. Entrena, evalua y exporta videos.

## Checkpoints

Los checkpoints y artefactos se guardan en:

- `artifacts/lunarlander/checkpoints`
- `artifacts/assault/checkpoints`

Para reanudar entrenamiento, carga el ultimo checkpoint en el notebook. Cada notebook incluye un path de carga y un archivo `config.json`.

## Artefactos

- `artifacts/*/logs` para TensorBoard
- `artifacts/*/videos` para videos de evaluacion

## Notas

Solo se usan los siguientes algoritmos: DQN, DQN+PER.
