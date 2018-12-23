#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:27:29 2018

@author: travisbarton
"""


'''


PHYS


'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(physpreds))
confm
plot_confusion_matrix(confm, ['Not Physics', 'Physics'], normalize = True, title = "Is Physics?")



'''


BIO



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(biopreds))
confm
plot_confusion_matrix(confm, ['Not Bio', 'Bio'], normalize = True, title = "Is Bio?")




'''


Med



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(medpreds))
confm
plot_confusion_matrix(confm, ['Not Med', 'Med'], normalize = True, title = "Is Med?")





'''


GEO



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(geopreds))
confm
plot_confusion_matrix(confm, ['Not Geo', 'Geo'], normalize = True, title = "Is Geo?")





'''


CHEM



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(chempreds))
confm
plot_confusion_matrix(confm, ['Not Chem', 'Chem'], normalize = True, title = "Is Chem?")



'''


Astro



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(astropreds))
confm
plot_confusion_matrix(confm, ['Not Astro', 'Astro'], normalize = True, title = "Is Astro?")


'''


Other



'''


confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(otherpreds))
confm
plot_confusion_matrix(confm, ['Not Other', 'Other'], normalize = True, title = "Is Other?")


































