"""Common constants used when training agents to complete tasks in iTHOR, the
interactive version of AI2-THOR."""

from collections import OrderedDict
from typing import Set, Dict
import itertools
import os

# THOR_COMMIT_ID = "b302ab97ccd30dba620fe84b1733f5be3e8558fc"
THOR_COMMIT_ID = "f0825767cd50d69f666c7f282e54abfe58f1e917"

MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_DOWN = "LookDown"
LOOK_UP = "LookUp"
END = "End"
DIRECTIONAL_AHEAD_PUSH = "DirectionalPush"
DIRECTIONAL_BACK_PUSH = "DirectionalPush"
DIRECTIONAL_RIGHT_PUSH = "DirectionalPush"
DIRECTIONAL_LEFT_PUSH = "DirectionalPush"
MOVE_BACK = "MoveBack"
PICK_UP = "PickupObject"
DROP = "DropHandObject"

VISIBILITY_DISTANCE = 1.25
FOV = 90.0

ORDERED_SCENE_TYPES = ("kitchens", "livingrooms", "bedrooms", "bathrooms")

NUM_SCENE_TYPES = len(ORDERED_SCENE_TYPES)

obstacle_types = ['ArmChair', 'DogBed', 'Box', 'Chair', 'Desk', 'DiningTable', 'SideTable', 'Sofa', 'Stool', 'Television', 'Pillow', 'Bread', 'Apple', 'AlarmClock', 'Lettuce', 'GarbageCan', 'Laptop', 'Microwave', 'Pot', 'Tomato']


def make_scene_name(type_ind, scene_num):
    if type_ind == 1:
        return "FloorPlan" + str(scene_num) + "_physics"
    elif scene_num < 10:
        return "FloorPlan" + str(type_ind) + "0" + str(scene_num) + "_physics"
    else:
        return "FloorPlan" + str(type_ind) + str(scene_num) + "_physics"


SCENES_TYPE_TO_SCENE_NAMES = OrderedDict(
    [
        (
            ORDERED_SCENE_TYPES[type_ind - 1],
            tuple(
                make_scene_name(type_ind=type_ind, scene_num=scene_num)
                for scene_num in range(1, 31)
            ),
        )
        for type_ind in range(1, NUM_SCENE_TYPES + 1)
    ]
)

SCENES_TYPE_TO_TRAIN_SCENE_NAMES = OrderedDict(
    (key, scenes[:20]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

SCENES_TYPE_TO_VALID_SCENE_NAMES = OrderedDict(
    (key, scenes[20:25]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

SCENES_TYPE_TO_TEST_SCENE_NAMES = OrderedDict(
    (key, scenes[25:30]) for key, scenes in SCENES_TYPE_TO_SCENE_NAMES.items()
)

ALL_SCENE_NAMES = sum(SCENES_TYPE_TO_SCENE_NAMES.values(), tuple())

TRAIN_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_TRAIN_SCENE_NAMES.values()), tuple()
)

VALID_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_VALID_SCENE_NAMES.values()), tuple()
)
TEST_SCENE_NAMES = sum(
    (scenes for scenes in SCENES_TYPE_TO_TEST_SCENE_NAMES.values()), tuple()
)

TRAIN_SCENE_NAMES_SET = set(TRAIN_SCENE_NAMES)
VALID_SCENE_NAMES_SET = set(VALID_SCENE_NAMES)
TEST_SCENE_NAMES_SET = set(TEST_SCENE_NAMES)

_object_type_and_location_tsv = """
AlarmClock	bedrooms
Apple	kitchens
ArmChair	livingrooms,bedrooms
BaseballBat	bedrooms
BasketBall	bedrooms
Bathtub	bathrooms
BathtubBasin	bathrooms
Bed	bedrooms
Blinds	kitchens,bedrooms
Book	kitchens,livingrooms,bedrooms
Boots	livingrooms,bedrooms
Bottle	kitchens
Bowl	kitchens,livingrooms,bedrooms
Box	livingrooms,bedrooms
Bread	kitchens
ButterKnife	kitchens
Cabinet	kitchens,livingrooms,bedrooms,bathrooms
Candle	livingrooms,bathrooms
Cart	bathrooms
CD	bedrooms
CellPhone	kitchens,livingrooms,bedrooms
Chair	kitchens,livingrooms,bedrooms
Cloth	bedrooms,bathrooms
CoffeeMachine	kitchens
CoffeeTable	livingrooms,bedrooms
CounterTop	kitchens,livingrooms,bedrooms,bathrooms
CreditCard	kitchens,livingrooms,bedrooms
Cup	kitchens
Curtains	kitchens,livingrooms,bedrooms
Desk	bedrooms
DeskLamp	livingrooms,bedrooms
DiningTable	kitchens,livingrooms,bedrooms
DishSponge	kitchens,bathrooms
Drawer	kitchens,livingrooms,bedrooms,bathrooms
Dresser	livingrooms,bedrooms,bathrooms
Egg	kitchens
Faucet	kitchens,bathrooms
FloorLamp	livingrooms,bedrooms
Footstool	bedrooms
Fork	kitchens
Fridge	kitchens
GarbageCan	kitchens,livingrooms,bedrooms,bathrooms
HandTowel	bathrooms
HandTowelHolder	bathrooms
HousePlant	kitchens,livingrooms,bedrooms,bathrooms
Kettle	kitchens
KeyChain	livingrooms,bedrooms
Knife	kitchens
Ladle	kitchens
Laptop	kitchens,livingrooms,bedrooms
LaundryHamper	bedrooms
LaundryHamperLid	bedrooms
Lettuce	kitchens
LightSwitch	kitchens,livingrooms,bedrooms,bathrooms
Microwave	kitchens
Mirror	kitchens,livingrooms,bedrooms,bathrooms
Mug	kitchens,bedrooms
Newspaper	livingrooms
Ottoman	livingrooms,bedrooms
Painting	kitchens,livingrooms,bedrooms,bathrooms
Pan	kitchens
PaperTowel	kitchens,bathrooms
Pen	kitchens,livingrooms,bedrooms
Pencil	kitchens,livingrooms,bedrooms
PepperShaker	kitchens
Pillow	livingrooms,bedrooms
Plate	kitchens,livingrooms
Plunger	bathrooms
Poster	bedrooms
Pot	kitchens
Potato	kitchens
RemoteControl	livingrooms,bedrooms
Safe	kitchens,livingrooms,bedrooms
SaltShaker	kitchens
ScrubBrush	bathrooms
Shelf	kitchens,livingrooms,bedrooms,bathrooms
ShowerCurtain	bathrooms
ShowerDoor	bathrooms
ShowerGlass	bathrooms
ShowerHead	bathrooms
SideTable	livingrooms,bedrooms
Sink	kitchens,bathrooms
SinkBasin	kitchens,bathrooms
SoapBar	bathrooms
SoapBottle	kitchens,bathrooms
Sofa	livingrooms,bedrooms
Spatula	kitchens
Spoon	kitchens
SprayBottle	bathrooms
Statue	kitchens,livingrooms,bedrooms
StoveBurner	kitchens
StoveKnob	kitchens
TeddyBear	bedrooms
Television	livingrooms,bedrooms
TennisRacket	bedrooms
TissueBox	livingrooms,bedrooms,bathrooms
Toaster	kitchens
Toilet	bathrooms
ToiletPaper	bathrooms
ToiletPaperHanger	bathrooms
Tomato	kitchens
Towel	bathrooms
TowelHolder	bathrooms
TVStand	livingrooms
Vase	kitchens,livingrooms,bedrooms
Watch	livingrooms,bedrooms
WateringCan	livingrooms
Window	kitchens,livingrooms,bedrooms,bathrooms
WineBottle	kitchens
"""

OBJECT_TYPE_TO_SCENE_TYPES = OrderedDict()
for ot_tab_scene_types in _object_type_and_location_tsv.split("\n"):
    if ot_tab_scene_types != "":
        ot, scene_types_csv = ot_tab_scene_types.split("\t")
        OBJECT_TYPE_TO_SCENE_TYPES[ot] = tuple(sorted(scene_types_csv.split(",")))

SCENE_TYPE_TO_OBJECT_TYPES: Dict[str, Set[str]] = OrderedDict(
    ((k, set()) for k in ORDERED_SCENE_TYPES)
)
for ot_tab_scene_types in _object_type_and_location_tsv.split("\n"):
    if ot_tab_scene_types != "":
        ot, scene_types_csv = ot_tab_scene_types.split("\t")
        for scene_type in scene_types_csv.split(","):
            SCENE_TYPE_TO_OBJECT_TYPES[scene_type].add(ot)


OTHER_PALETTE= [
    1.0, 1.0, 1.0, # out-of-bounds (or no category)
    0.9490196078, 0.9490196078, 0.9490196078, # Floor (explored 视野看到过的区域)
    0.3450980392, 0.3843137255, 0.4352941176, # Wall (with no category) in Obstacle
    0.9019607843, 0.2745098039, 0.1607843137, # Visited (self.loc_on_map)
]
MASS_LIST = ['Candle', 'SoapBottle', 'ToiletPaper', 'SoapBar', 'SprayBottle', 'TissueBox', 'DishSponge', 'PaperTowelRoll', 'Book', 'CreditCard', 'Dumbbell', 'Pen', 'Pencil', 'CellPhone', 'Laptop', 'CD', 'AlarmClock', 'Statue', 'Mug', 'Bowl', 'TableTopDecor', 'Box', 'RemoteControl', 'Vase', 'Watch', 'Newspaper', 'Plate', 'WateringCan', 'Fork', 'PepperShaker', 'Spoon', 'ButterKnife', 'Pot', 'SaltShaker', 'Cup', 'Spatula', 'WineBottle', 'Knife', 'Pan', 'Ladle', 'Egg', 'Kettle', 'Bottle', 'Drawer', 'Toilet', 'ShowerCurtain', 'ShowerDoor', 'Cabinet', 'Blinds', 'LaundryHamper', 'Safe', 'Microwave', 'Fridge']
# id starts from 1
# 共115类（原本117类，去除了Mirror, ShowerGlass）（选了59类）(13+3 -> 59+3)
SEGMENTATION_CATEGORY_to_COLOR= {
    "Book": [0.8944901560798098,0.5054300021592404,0.6496021889514848],
    "Bottle": [0.9908287233229586,0.7382462321618167,0.9524192408378547],
    "Knife": [0.4349903395873266,0.4117905139436565,0.6912039810021936],
    "Bread": [0.4365492147755386,0.9948515419446143,0.4214773501002491],
    "Fork": [0.4222005467330625,0.4967514325534182,0.9992319513428765],
    "Potato": [0.4347986766613366,0.6245784954282467,0.5076455237673995],
    "SoapBottle": [0.4224810584826086,0.7770041528105601,0.6899329698785336],
    "Kettle": [0.42658251448533285,0.4827443454982429,0.5315165480690832],
    "Pan": [0.5075438267472905,0.7729686159575577,0.9984499021218259],
    "Plate": [0.8422093755460658,0.5881006020135334,0.47051315951951933],
    "Tomato": [0.5620349637498508,0.9988460741891618,0.8139211067995725],
    "Vase": [0.6332702585633582,0.8935690151294701,0.437824116288111],
    "Egg": [0.9823561477815355,0.8413183275638974,0.8245917332898306],
    "CreditCard": [0.7811330486149806,0.5357914023658132,0.7836437886679496],
    "WineBottle": [0.8257169975621231,0.7105799405189296,0.9894094998481406],
    "Pot": [0.772176689014993,0.4286419562505512,0.5884091000687061],
    "Spatula": [0.42269259796225633,0.8259742319357074,0.8383484305646023],
    "PaperTowelRoll": [0.9810781166501736,0.7864736564357785,0.4573066855581681],
    "Cup": [0.8028347814481684,0.6702329318905331,0.7523674551401582],
    "Bowl": [0.4742696522804535,0.636080659081543,0.6585047224692212],
    "SaltShaker": [0.47619102891830567,0.47128239241808145,0.9320638157848696],
    "PepperShaker": [0.4658916707182335,0.5955276002539428,0.7951815544544812],
    "Lettuce": [0.4983744674686366,0.5192138085970958,0.41315060130903636],
    "ButterKnife": [0.6095506399681903,0.6239866408725534,0.9983342582245406],
    "Apple": [0.9810794729630511,0.7579775718565557,0.7062171615114837],
    "DishSponge": [0.5049878001783026,0.6928625923533266,0.4159809140396412],
    "Spoon": [0.8311310526501772,0.5946517762175111,0.9965698018908103],
    "Mug": [0.9806857122013507,0.6565727200237373,0.8458024410301995],
    "Statue": [0.6507613973761683,0.8910763537267937,0.8991005754668334],
    "Ladle": [0.4260889924630301,0.5000461733226053,0.743523191814956],
    "CellPhone": [0.7480631716111666,0.44740100490626933,0.9881769323543009],
    "Pen": [0.7774394574879303,0.43331359452124396,0.838136453089127],
    "SprayBottle": [0.8112887471120989,0.9092190375358387,0.9100413100045698],
    "Pencil": [0.8610686991790012,0.4231923535406448,0.7117582969635872],
    "Box": [0.41801872456254524,0.6375571442651786,0.8950179517163757],
    "Laptop": [0.41702856221022794,0.5645209753865156,0.5908929593881218],
    "TissueBox": [0.7423155117747713,0.6171303846726052,0.8741127616033184],
    "KeyChain": [0.43400960531095767,0.8885562155534982,0.463331490737266],
    "Pillow": [0.4940184057786778,0.41680806842382645,0.8185955094774747],
    "RemoteControl": [0.9764180791626699,0.4254419870800033,0.9036065292861256],
    "Watch": [0.7017563703100366,0.9469395708020619,0.460354068059876],
    "Newspaper": [0.791469318752095,0.7839543076976592,0.6915128175990064],
    "WateringCan": [0.6857346288073977,0.41278468313502,0.9805018866141503],
    "Boots": [0.4314531627716158,0.4391313955085093,0.9134022766540085],
    "Candle": [0.6164628069436479,0.4248834529763948,0.5004373088294808],
    "BaseballBat": [0.42119903075029963,0.9999095283304374,0.9874875143685572],
    "BasketBall": [0.4276839398512328,0.9714039088485011,0.7609383379001484],
    "AlarmClock": [0.6870041548518976,0.5592505913046075,0.6323919807212712],
    "CD": [0.42320504249577656,0.7052331700958941,0.5304096637584581],
    "TennisRacket": [0.6198199107330338,0.7511121996012294,0.9292339652333419],
    "TeddyBear": [0.4124591088722159,0.8620126057272729,0.7552076550411441],
    "Cloth": [0.683470354851001,0.4712806260351029,0.48647257471091254],
    "Dumbbell": [0.9968164124530252,0.5578881538424664,0.7421338192704278],
    "Towel": [0.859027176281372,0.7743787721070794,0.4527472983599562],
    "HandTowel": [0.7087265789014027,0.6364285726228833,0.706574858019122],
    "Plunger": [0.9965984216572693,0.8513707452926766,0.991086256960618],
    "SoapBar": [0.7459261914342274,0.4255415321215695,0.905804199025346],
    "ToiletPaper": [0.9820678816450995,0.500854542890096,0.5846970468057812],
    "ScrubBrush": [0.607386105213221,0.8194975433468026,0.999535049282008],

    # # 暂时筛掉了一部分不作分割
    "Painting": [0.44482081498245385,0.589234344368492,0.9920596434258766],
    "Fridge": [0.4495484080246552,0.8001285735320048,0.5078423551004748],
    "GarbageCan": [0.9859636353689167,0.541131079383881,0.9161773072034288],
    "GarbageBag": [0.6444225361394159,0.4644568465779099,0.7075077558463427],
    "StoveBurner": [0.9481562984664261,0.8629915451676468,0.5285916939054338],
    "Drawer": [0.42506379016700147,0.9249332014421527,0.9724808257064019],
    "CounterTop": [0.8232982188451926,0.4218597323952243,0.4413601947822409],
    "Cabinet": [0.4138288080596801,0.4122452236696893,0.833440360314779],
    "StoveKnob": [0.6117716459227811,0.8589885473677683,0.7510209593744287],
    "Window": [0.9894536145512085,0.6496813320510602,0.44623719817683605],
    "Sink": [0.43105225189174357,0.7081255646268312,0.8437023299929306],
    "Microwave": [0.8711431947905237,0.42047936512769485,0.9608891984277376],
    "Shelf": [0.9792291014256156,0.6469733248587848,0.6635725548897035],
    "HousePlant": [0.9450210211102331,0.9890015161458519,0.4639865407534175],
    "Toaster": [0.6480994607420907,0.4132240324165712,0.8799256004406653],
    "CoffeeMachine": [0.626316470064064,0.9950065584809691,0.9993820725841043],
    "SinkBasin": [0.6376577700884991,0.8381874200265806,0.5844963729955649],
    "LightSwitch": [0.593104731949016,0.9683565926484853,0.6552071269594395],
    "ShelvingUnit": [0.6252313272032415,0.678886410331077,0.5962794182342921],
    "Stool": [0.7121337407271131,0.7628566185446412,0.4732130855457434],
    "Faucet": [0.9875329955477997,0.7306092229749234,0.5782881733988047],
    "Chair": [0.9451777667556885,0.4380305906734387,0.5436577665300555],
    "SideTable": [0.737272098419955,0.7154782282617212,0.8553784919343072],
    "DiningTable": [0.9911722561733476,0.4756911138623119,0.8028949369217324],
    "Curtains": [0.4934367406343782,0.4166625294199333,0.4308172507937298],
    "Blinds": [0.9998562460456895,0.5776710254143145,0.5673046192680252],
    "Safe": [0.9330385893613959,0.4540038871112912,0.4144828219105657],
    "Television": [0.8696487773496814,0.8000637286517079,0.8165771815867713],
    "FloorLamp": [0.6985285438282378,0.5582143051024014,0.5292814358946653],
    "DeskLamp": [0.6094013084551588,0.7684320881969334,0.644337317775211],
    "ArmChair": [0.5573071044546607,0.5819763188738623,0.41809198282305327],
    "CoffeeTable": [0.9974698568271696,0.8714352610214479,0.661345876498487],
    "TVStand": [0.44673383374821296,0.8961910618875646,0.6470804195430805],
    "Sofa": [0.5478013395044111,0.42857497869891686,0.6573061724842236],
    "Ottoman": [0.8178470416682868,0.983901843980427,0.7788059247107461],
    "Desk": [0.8532583427919268,0.6213816004914722,0.6260963307004267],
    "Dresser": [0.6517501653211388,0.5704452308521969,0.8877933405028609],
    "DogBed": [0.5895803588667453,0.8347957406384718,0.8668112483679996],
    "RoomDecor": [0.8166571702022528,0.8012179040768824,0.9895768481402472],
    "Bed": [0.5751086388357423,0.6797347505594714,0.7688532553550423],
    "Poster": [0.6364145256074656,0.42740454914764986,0.6027294980884664],
    "LaundryHamper": [0.424137847936713,0.4248061888566223,0.5860366035904963],
    "TableTopDecor": [0.7967841092008857,0.9785389384535714,0.4550830488129234],
    "Desktop": [0.8062979215979594,0.8801508326529172,0.7368443141576313],
    "Footstool": [0.6452974356193215,0.7162073785338885,0.4368216382486421],
    "BathtubBasin": [0.4167858918623678,0.5598298224989745,0.8804498561953907],
    "ShowerCurtain": [0.7863312015882292,0.8473639469240355,0.5746174865648033],
    "ShowerHead": [0.7051884693835773,0.7081296872376964,0.538110809682683],
    "Bathtub": [0.9866363135515801,0.6329913575256564,0.9465407075512579],
    "TowelHolder": [0.8353132116455175,0.7449711616036437,0.5563221213843866],
    "ToiletPaperHanger": [0.41390658706644445,0.7405237804372898,0.9221348744908006],
    "HandTowelHolder": [0.45437910804571274,0.5457007940402893,0.49609003002224766],
    "Toilet": [0.7541033654977085,0.8942841455881726,0.9804712500332733],
    "ShowerDoor": [0.8443715903355739,0.866797623016698,0.43731629999247135],
    "AluminumFoil": [0.5767097549168565,0.6005670836280637,0.7570740261553073],
    "VacuumCleaner": [0.9971358530504674,0.9121882032734007,0.7363162215902606]
}
CATEGORY_PALETTE = list(itertools.chain(*SEGMENTATION_CATEGORY_to_COLOR.values()))