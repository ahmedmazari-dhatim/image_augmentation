from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import parameters as iap
from imgaug.parameters import StochasticParameter, Deterministic
from imgaug import augmenters as iaa


from scipy import ndimage, misc

class BinomialRows(StochasticParameter):
    def __init__(self, p):
        super(BinomialRows, self).__init__()

        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
        h, w, c = size
        drops = random_state.binomial(1, p, (h, 1, c))
        drops_rows = np.tile(drops, (1, w, 1))
        return drops_rows

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "BinomialRows(%.4f)" % (self.p,)
        else:
            return "BinomialRows(%s)" % (self.p,)

class BinomialColumns(StochasticParameter):
    def __init__(self, p):
        super(BinomialColumns, self).__init__()

        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
        h, w, c = size
        drops = random_state.binomial(1, p, (1, w, c))
        drops_columns = np.tile(drops, (h, 1, 1))
        return drops_columns

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "BinomialColumns(%.4f)" % (self.p,)
        else:
            return "BinomialColumns(%s)" % (self.p,)



p_row = 0.2
p_column=0.1
aug = iaa.Sequential([
    iaa.Dropout(p=BinomialRows(1-p_row)),
    iaa.Dropout(p=BinomialColumns(1-p_column)),
])

image = ndimage.imread("/home/ahmed/Downloads/test/brut_image/number.png")
new_image=aug.augment_image(image)


#misc.imshow(aug.augment_image(image))