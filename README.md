Author: Berangere Villatte, Université de Montréal

This projects aims to analyse and generate plots from Heart Rate Variability data
obtained with an ECG (AUDACE Project)

HRV data were generated using house scripts in Matlab
########
Run hrv.py

This scripts generates:
1. Linear mixed effects model
2. Compute supervised and unsupervised automatic learning from subsets of df
    
    Supervised:   KNN with k-fold cross-validation 
                  and a comparison with train_test_split() method
    
    Unsupervised: PCA (Principal Component Analysis)
    
3. Plot algorithm results 


Data informations (in French):


Base de données : Données physiologiques (ECG) obtenues sur des participants sains pendant 3 tâches de stress (Mental, Noise, CPT) et 3 périodes de repos
Échantillon : N = 30; 15F et 15H
Variables		: 	
					1. Participant 							 	Facteur, 30 niveaux

					2. Sex 									Variable indépendante, inter-groupe
															type: Facteur, 2 niveaux (H, F)

					3. Task									Variable indépendante, intra-groupe
															type: facteur, 6 niveaux (BASELINEPRE_MENTAL_TSK, MENTAL_TSK, 
																		BASELINEPRE_NOSIE_TSK, NOISE_TSK, 
																		BASELINEPRE_CPT_TSK, CPT_TSK)

					4. Feature		 						Variable indépendante, intra-groupe
															type: facteur, 12 niveaux(MeanNN, RMSSD, SDNN, HF, LF, MaxNN, MaxNN_smoothed0p15, 
																		MinNN, MinNN_smoothed0p15, PeakHF, PeakLF, RatioLFHF)

					
					5. pChange								Variable dépendante continue, Série temporelle
															type: float() 

					6. Time									Variable indépendante, intra-groupe
															type: facteur ordonné, 
																15 niveaux(t00, t01, t02, t03, t04, t05, 
																	t06, t07, t08, t09, t10, t11, t12, t13, t14)


Autres variables non-utilisées de la base de données
	SegmentStartRefTaskOnset				Variable continue, type: int()		(devrait être)
	SegmentStartRefTaskOnset				Variable continue, type: int()		(devrait être)
	Value						Variable dépendante continue, type: float()
	segDur						Variable continue, type: int()

Objectif:
1) Déterminer si le temps à un effet sur les valeurs (pChange) des paramètres HRV (Feature) pour le RMSSD, SDNN et MeanNN pour chaque tâche individuellement.
2) Déterminer si les valeurs (pChange) des paramètres HRV (Feature) au cours du temps est différent selon le sexe 

Analyses: 
1) Série temporelle: déterminer si les résidus sont normalement distribués 
	Shapiro-wilk, p > 0.05 pour que la distribution des résidus soit normale

2) PLAN A 
	Si oui: ANOVA mixte, 	intra-groupe 1: temps (Time)
				inter-groupe 2: sexe (Sex)
				Post-hoc: 1v2, correction: Holm

	H1: interaction sexe/temps pour le CPT seulement, pour MeanNN et RMSSD
	H2: effet principal de temps pour MeanNN, RMSSD et SDNN pour les 3 tâches (MENTAL, NOISE, CPT)
	H3: effet principal de sexe pour le RMSSD et CPT




3) PLAN B
	Si non: Modèle linéaire mixte (linear mixte effects)
					effets fixes : pChange ~ temps (Time) * Sexe (Sex)
					effets aléatoires : ~1 | Participant

					ANOVA mixte avec le modèle

					Post-hoc: 1v2, comparaisons inter/intra/interintra-groupe: Chi2

	H1: interaction sexe/temps pour le CPT seulement, pour MeanNN et RMSSD
	H2: effet principal de temps pour MeanNN, RMSSD et SDNN pour les 3 tâches (MENTAL, NOISE, CPT)
	H3: effet principal de sexe pour le RMSSD et CPT


Grahiques: 
panel grid 3 x 3 		: Task (3) x Feat (3)
2 courbes 			: 1 par sexe (H, F)
plot				: mean(pChange)
SD 					: sd(pChange) au niveau du groupe
Signif				: barres de signif, 1 par effet principal et 1 par interaction





