from sklearn.preprocessing import StandardScaler


def scale_vectors(vectors: list) -> list:
    PredictorScaler = StandardScaler()
    PredictorScalerFit = PredictorScaler.fit(vectors)
    return PredictorScalerFit.transform(vectors)
