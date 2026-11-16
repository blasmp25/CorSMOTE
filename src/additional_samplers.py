"""
Samplers adicionales para comparar contra CorSMOTE.

Se descartaron SMOTE-Cov (Leguen de Varona et al., 2020) y FW-SMOTE
(Maldonado, Vairetti, Fernandez, Herrera, 2021/22) porque no existe
codigo publico de los autores en ningun sitio (se buscó en GitHub,
paginas personales/institucionales de los autores, el repositorio de
software del grupo SCI2S de Herrera -que si publica codigo de la mayoria
de sus otros papers- y digibug.ugr.es -que solo tiene el PDF). Usar una
reimplementacion propia sin poder contrastarla contra el codigo original
no es fiable para una comparacion experimental seria, asi que se
eliminan del estudio.

Se mantienen Safe-Level-SMOTE, SMOTE-IPF, y MDO (Mahalanobis Distance-based
Oversampling, Abdi & Hashemi), via la libreria `smote_variants` (Kovacs,
Neurocomputing 2019, pip install smote-variants), que SI esta publicada,
revisada, y con codigo verificable. MDO sustituye a SMOTE-Cov como baseline
de "correlacion/covarianza real entre atributos": genera muestras sinteticas
que preservan la distancia de Mahalanobis (basada en la matriz de covarianza
de la clase minoritaria) en vez de interpolar por vecinos.

IMPORTANTE sobre MDO: confirma la referencia bibliografica exacta antes de
citarlo (Abdi & Hashemi tienen mas de una publicacion sobre oversampling;
verifica en Google Scholar cual es la que corresponde al metodo MDO
implementado en smote_variants).

IMPORTANTE: `smote_variants` expone `.sample(X, y)`, no `.fit_resample(X, y)`
(la interfaz que espera imblearn.pipeline.Pipeline), de ahi el wrapper.
Antes de fijar el grid de hiperparametros, confirma los nombres reales del
constructor en tu entorno:
    import smote_variants as sv
    help(sv.Safe_Level_SMOTE.__init__)
    help(sv.SMOTE_IPF.__init__)
Los nombres usados abajo ('n_neighbors', 'n_folds') son mi mejor estimacion,
no una confirmacion al 100%.
"""

import numpy as np
from sklearn.base import BaseEstimator


class SVSamplerWrapper(BaseEstimator):
    """
    Envuelve un oversampler de `smote_variants` (interfaz .sample(X, y))
    para que sea compatible con imblearn.pipeline.Pipeline (interfaz
    .fit_resample(X, y)).

    Uso:
        import smote_variants as sv
        wrapper = SVSamplerWrapper(sv.Safe_Level_SMOTE, n_neighbors=5)
        pipe = Pipeline([('sampler', wrapper), ('clf', rf)])
        # GridSearchCV con: {'sampler__n_neighbors': [3, 5, 7, 9]}
    """

    def __init__(self, sv_class, random_state=None, **sv_kwargs):
        self.sv_class = sv_class
        self.random_state = random_state
        for k, v in sv_kwargs.items():
            setattr(self, k, v)
        self._sv_kwarg_names = list(sv_kwargs.keys())

    def get_params(self, deep=True):
        params = {"sv_class": self.sv_class, "random_state": self.random_state}
        for name in self._sv_kwarg_names:
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
            if k not in ("sv_class", "random_state") and k not in self._sv_kwarg_names:
                self._sv_kwarg_names.append(k)
        return self

    def fit_resample(self, X, y):
        kwargs = {name: getattr(self, name) for name in self._sv_kwarg_names}
        if self.random_state is not None:
            kwargs.setdefault("random_state", self.random_state)
        oversampler = self.sv_class(**kwargs)
        X_samp, y_samp = oversampler.sample(np.asarray(X), np.asarray(y))
        return X_samp, y_samp
