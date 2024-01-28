import os

from nlp_diplomacy.modulos import exploratory_analysis,model_LogisticRegression, model_LSTM
from orquestador2.orquestador2 import Orchestrator


if not os.path.exists( "logs/" ):
	print("Directorio de log no creado, Creándolo ........")
	os.mkdir("logs/")

steps = [exploratory_analysis()]

orquestador = Orchestrator('nlp', steps)
orquestador.ejecutar()