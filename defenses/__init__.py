from .secalign import *
from .secalign.defense_secalign import secalign_batch
from .attentiontracker import *
from .attentiontracker.defense_attentiontracker import attentiontracker_batch
from .pisanitizer import *
from .pisanitizer.defense_pisanitizer import pisanitizer_batch
from .datasentinel import *
from .datasentinel.defense_datasentinel import datasentinel_batch
from .promptguard import *
from .promptguard.defense_promptguard import promptguard_batch
from .promptarmor import *
from .promptarmor.defense_promptarmor import promptarmor_batch
from .datafilter import *
from .piguard import *
from .piguard.defense_piguard import piguard_batch
from .no_defense import *
from .no_defense import no_defense_batch

DEFENSES = {
    "secalign": secalign,
    "attentiontracker": attentiontracker,
    "pisanitizer": pisanitizer,
    "datasentinel": datasentinel,
    "promptguard": promptguard,
    "promptarmor": promptarmor,
    "datafilter": datafilter,
    "piguard": piguard,
    "none": no_defense,  # Alias for convenience
    "sandwich": sandwich,
    "instructional": instructional,
}

# 支持批量处理的 defense 函数
DEFENSES_BATCH = {
    "datafilter": datafilter_batch,
    "datasentinel": datasentinel_batch,
    "secalign": secalign_batch,
    "promptguard": promptguard_batch,
    "promptarmor": promptarmor_batch,
    "attentiontracker": attentiontracker_batch,
    "piguard": piguard_batch,
    "pisanitizer": pisanitizer_batch,
    "none": no_defense_batch,
    "sandwich": sandwich_batch,
    "instructional": instructional_batch,
}

