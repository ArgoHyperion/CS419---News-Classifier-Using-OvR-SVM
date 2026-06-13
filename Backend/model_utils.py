import cvxopt
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
cvxopt.solvers.options["show_progress"] = False

import re
import spacy
from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
stop_words.discard("us")

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser"]
)

JUNK_TOKENS = {
    "'s",
}


def preprocess_text(text):
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    person_tokens = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for token in ent:
                person_tokens.add(token.text)

    processed = []
    prev_person = False

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        is_person = (
            token.ent_type_ == "PERSON"
            or token.text in person_tokens
            or "'" in token.text
        )

        if is_person:
            if not prev_person:
                processed.append("PERSON")
            prev_person = True
            continue
        prev_person = False

        CURRENCY_SYMBOLS = {"$", "£", "€"}
        MAGNITUDE_SUFFIXES = {"m", "bn", "b", "k"}

        if token.text in CURRENCY_SYMBOLS:
            processed.append("MONEY")
            continue

        if re.match(
                r"^\d+(?:\.\d+)?(?:bn|m|b|k)$",
                token.text,
                re.I
        ):
            processed.append("NUM")
            continue

        if token.text.lower() in MAGNITUDE_SUFFIXES:
            continue

        if token.like_num:
            processed.append("NUM")
            continue

        lemma = token.lemma_.strip().lower()
        if not lemma:
            continue
        if lemma in JUNK_TOKENS:
            continue
        if lemma in stop_words:
            continue
        if not re.search(r"[a-z0-9]", lemma):
            continue
        if len(lemma) < 2:
            continue
        processed.append(lemma)

    return processed

class TfidfVectorizer(CountVectorizer, TransformerMixin):

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def _compute_idf(self, X):

        n_samples = X.shape[0]

        df = np.bincount(
            X.indices,
            minlength=X.shape[1]
        )

        if self.smooth_idf:
            idf = np.log(
                (1.0 + n_samples) / (1.0 + df)
            ) + 1.0
        else:
            idf = np.log(
                n_samples / df
            ) + 1.0

        return idf.astype(np.float64)

    def fit(self, raw_documents, y=None):

        X = super().fit_transform(raw_documents)

        if self.use_idf:
            self.idf_ = self._compute_idf(X)
        else:
            self.idf_ = np.ones(
                X.shape[1],
                dtype=np.float64
            )

        return self

    def fit_transform(self, raw_documents, y=None):

        X = super().fit_transform(raw_documents)
        X = X.astype(np.float64)

        if self.sublinear_tf:
            X.data = np.log(X.data) + 1.0

        if self.use_idf:
            self.idf_ = self._compute_idf(X)
            X = X.multiply(self.idf_)

        if self.norm is not None:
            X = normalize(
                X,
                norm=self.norm,
                copy=False
            )

        return X

    def transform(self, raw_documents):

        X = super().transform(raw_documents)
        X = X.astype(np.float64)

        if self.sublinear_tf:
            X.data = np.log(X.data) + 1.0

        if self.use_idf:
            X = X.multiply(self.idf_)

        if self.norm is not None:
            X = normalize(
                X,
                norm=self.norm,
                copy=False
            )

        return X

    @property
    def idf_(self):
        return self._idf

    @idf_.setter
    def idf_(self, value):
        self._idf = np.asarray(
            value,
            dtype=np.float64
        )

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)

    def inverse_transform(self, X):
        return super().inverse_transform(X)


class ChiSquareSelector(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, k=10000):
        self.k = k

    def fit(self, X, y):
        X_bin = (X > 0).astype(np.int8)
        classes = np.unique(y)
        scores = np.zeros(X.shape[1])

        for cls in classes:
            positive = (y == cls)
            negative = ~positive

            A = np.asarray(
                X_bin[positive].sum(axis=0)
            ).ravel()
            B = np.asarray(
                X_bin[negative].sum(axis=0)
            ).ravel()
            C = positive.sum() - A
            D = negative.sum() - B
            N = A + B + C + D
            numerator = (
                N * (A * D - B * C) ** 2
            )

            denominator = (
                (A + C)
                * (B + D)
                * (A + B)
                * (C + D)
            )
            chi = numerator / (
                denominator + 1e-12
            )

            scores += chi

        self.scores_ = scores
        self.selected_features_ = np.argsort(
            scores
        )[::-1][:self.k]
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            "selected_features_"
        )
        return X[:, self.selected_features_]

class BinarySVM(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        C=1.0,
        kernel="linear",
        gamma="scale",
        degree=3,
        coef0=0.0,
        class_weight=None,
        beta=0.99,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.beta = beta

    def _compute_gamma(self, X):
        n_features = X.shape[1]
        if self.gamma == "scale":
            X_dense = X.toarray()
            var = X_dense.var()
            if var == 0:
                return 1.0 / n_features
            return 1.0 / (n_features * var)
        if self.gamma == "auto":
            return 1.0 / n_features
        return float(self.gamma)

    def _get_kernel(self, X1, X2):
        if self.kernel == "linear":
            return (X1 @ X2.T).toarray()
        X1_d = X1.toarray()
        X2_d = X2.toarray()
        gamma = self.eff_gamma_
        if self.kernel == "rbf":
            sq1 = np.sum(X1_d ** 2, axis=1)[:, None]
            sq2 = np.sum(X2_d ** 2, axis=1)[None, :]
            dist = np.maximum(
                sq1 + sq2 - 2 * (X1_d @ X2_d.T),
                0.0,
            )
            return np.exp(-gamma * dist)
        if self.kernel == "poly":
            return (
                gamma * (X1_d @ X2_d.T)
                + self.coef0
            ) ** self.degree
        if self.kernel == "sigmoid":
            return np.tanh(
                gamma * (X1_d @ X2_d.T)
                + self.coef0
            )
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        y = np.where(y > 0, 1.0, -1.0)
        n_samples = X.shape[0]
        self.eff_gamma_ = self._compute_gamma(X)
        C_arr = np.full(
            n_samples,
            self.C,
            dtype=float,
        )
        if self.class_weight == "effective":
            classes, counts = np.unique(
                y,
                return_counts=True,
            )
            weights = (
                (1.0 - self.beta)  / (1.0 - np.power(self.beta, counts))
            )
            weights = (
                weights / weights.sum() * len(classes)
            )

            weight_map = dict(
                zip(classes, weights)
            )

            for i in range(n_samples):
                C_arr[i] *= weight_map[y[i]]

        K = self._get_kernel(X, X)
        P = cvxopt.matrix(
            np.outer(y, y) * K
            +
            1e-9 * np.eye(n_samples)
        )
        q = cvxopt.matrix(
            -np.ones(n_samples)
        )
        G = cvxopt.matrix(
            np.vstack([
                -np.eye(n_samples),
                np.eye(n_samples),
            ])
        )
        h = cvxopt.matrix(
            np.hstack([
                np.zeros(n_samples),
                C_arr,
            ])
        )

        A = cvxopt.matrix(
            y.reshape(1, -1)
        )

        b = cvxopt.matrix(0.0)

        try:
            solution = cvxopt.solvers.qp(
                P,
                q,
                G,
                h,
                A,
                b,
            )
            self.alpha_ = np.array(
                solution["x"]
            ).flatten()

        except Exception:
            self.alpha_ = np.zeros(
                n_samples,
                dtype=float,
            )

        sv_mask = self.alpha_ > 1e-6
        self.alpha_sv_ = self.alpha_[sv_mask]
        self.sv_ = X[sv_mask]
        self.sv_y_ = y[sv_mask]
        self.C_sv_ = C_arr[sv_mask]
        self.n_support_ = len(
            self.alpha_sv_
        )

        margin_mask = (
            self.alpha_sv_ > 1e-6
        ) & (
            self.alpha_sv_
            <
            self.C_sv_ - 1e-6
        )

        if np.any(margin_mask):
            K_margin = self._get_kernel(
                self.sv_[margin_mask],
                self.sv_,
            )
            self.b_ = np.mean(
                self.sv_y_[margin_mask]
                -
                K_margin @ (
                    self.alpha_sv_
                    *
                    self.sv_y_
                )
            )

        elif self.n_support_ > 0:
            K_sv = self._get_kernel(
                self.sv_,
                self.sv_,
            )
            self.b_ = np.mean(
                self.sv_y_
                -
                K_sv @ (
                    self.alpha_sv_
                    *
                    self.sv_y_
                )
            )

        else:
            self.b_ = 0.0
        return self

    def decision_function(self, X):
        check_is_fitted(
            self,
            [
                "sv_",
                "alpha_sv_",
                "b_",
            ],
        )

        if self.n_support_ == 0:
            return np.zeros(
                X.shape[0]
            )

        K = self._get_kernel(
            X,
            self.sv_,
        )

        return (K @ (self.alpha_sv_ * self.sv_y_) + self.b_
        )


class MultiClassSVM(
    BaseEstimator,
    ClassifierMixin,
):

    def __init__(
        self,
        C=1.0,
        kernel="linear",
        gamma="scale",
        degree=3,
        coef0=0.0,
        class_weight=None,
        beta=0.99,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.beta = beta

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        self.clfs_ = []

        for c in self.classes_:
            y_binary = np.where(
                y == c,
                1,
                -1,
            )

            clf = BinarySVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                class_weight=self.class_weight,
                beta=self.beta,
            )

            clf.fit(
                X,
                y_binary,
            )

            self.clfs_.append(
                clf
            )

        return self

    def decision_function(self, X):
        check_is_fitted(
            self,
            [
                "clfs_",
                "classes_",
            ],
        )

        return np.column_stack([
            clf.decision_function(X)
            for clf in self.clfs_
        ])

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[
            np.argmax(
                scores,
                axis=1,
            )
        ]

    def predict_confidence(self, X):
        scores = self.decision_function(X)
        return softmax(
            scores,
            axis=1,
        )