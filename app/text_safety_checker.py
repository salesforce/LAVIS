"""
    pip install nltk

    git clone https://github.com/clips/pattern.git
    cd pattern
    sed -i '140d' setup.py
    python setup.py install

    Usage:
        from text_safety_checker import handle_text

        handled_text = handle_text(original_text)
"""

import json
import nltk

nltk.download("omw-1.4")

from pattern.en import pluralize, singularize

# read text file line by line and return a list
def read_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


# write json file
def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


def neutralize_gender(text):
    # replace gendered words
    # also take care of pluralization
    tokens = text.split()

    tokens_neutral = []

    for i, token in enumerate(tokens):
        token = token.lower()

        # if it is plural, make it singular
        singular_form = singularize(token)
        is_plural = singular_form != token

        if singular_form in rplc:
            token_neutral = rplc[singular_form]
            if is_plural:
                token_neutral = pluralize(token_neutral)

            tokens_neutral.append(token_neutral)
        else:
            tokens_neutral.append(token)

    return " ".join(tokens_neutral)


def rm_nsfw(text):
    # remove nsfw words
    tokens = text.split()

    tokens_clean = []

    for token in tokens:
        if token not in blocklist:
            for i, bl in enumerate(blocklist):
                if bl in token:
                    break

            if i == len(blocklist) - 1:
                tokens_clean.append(token)
            else:
                tokens_clean.append("******")
        else:
            tokens_clean.append("******")

    return " ".join(tokens_clean)


def handle_text(text):
    text = neutralize_gender(text)
    text = rm_nsfw(text)

    return text


rplc = {
    "actor": "performer",
    "actress": "performer",
    "aunt": "pibling",
    "barman": "bartender",
    "barwoman": "bartender",
    "beautiful": "attractive",
    "boy": "child",
    "boyfriend": "partner",
    "brother": "sibling",
    "buddy": "friend",
    "businessman": "business executive",
    "businesswoman": "business executive",
    "cameraman": "camera operator",
    "camerawoman": "camera operator",
    "chairman": "chairperson",
    "chairwoman": "chairperson",
    "congressman": "legislator",
    "congresswoman": "legislator",
    "councilman": "councilperson",
    "councilwoman": "councilperson",
    "daughter": "child",
    "dude": "friend",
    "father": "parent",
    "female lawyer": "lawyer",
    "fireman": "firefighter",
    "firewoman": "firefighter",
    "freshman": "first-year student",
    "gentleman": "person",
    "girl": "child",
    "girlfriend": "partner",
    "grandfather": "grandparent",
    "grandmother": "grandparent",
    "handsome": "attractive",
    "he": "person",
    "her": "person",
    "hero": "heroix",
    "heroine": "heroix",
    "hers": "person's",
    "herself": "person",
    "him": "person",
    "himself": "person",
    "his": "person's",
    "hostess": "host",
    "househusband": "homemaker",
    "housewife": "homemaker",
    "husband": "partner",
    "lady": "person",
    "landlady": "landlord",
    "mailman": "mail carrier",
    "male lawyer": "lawyer",
    "man": "person",
    "man-made": "artificial",
    "mankind": "human beings",
    "manned": "crewed",
    "manpower": "workforce",
    "mother": "parent",
    "nephews": "nibling",
    "nieces": "nibling",
    "policeman": "police officer",
    "policewoman": "police officer",
    "postman": "mail carrier",
    "postwoman": "mail carrier",
    "repairman": "repairer",
    "repairwoman": "repairer",
    "salesman": "salesperson",
    "saleswoman": "salesperson",
    "she": "person",
    "sister": "sibling",
    "son": "child",
    "spokesman": "spokesperson",
    "spokeswoman": "spokesperson",
    "steward": "flight attendant",
    "stewardess": "flight attendant",
    "uncle": "pibling",
    "waiter": "waitperson",
    "waitress": "waitperson",
    "wife": "partner",
    "woman": "person",
    "workman": "worker",
    "workwoman": "worker",
}

blocklist = set(
    [
        "2 girls 1 cup",
        "2g1c",
        "4r5e",
        "5h1t",
        "5hit",
        "a$$",
        "a$$hole",
        "a_s_s",
        "a2m",
        "a54",
        "a55",
        "a55hole",
        "aeolus",
        "ahole",
        "alabama hot pocket",
        "alaskan pipeline",
        "anal",
        "anal impaler",
        "anal leakage",
        "analannie",
        "analprobe",
        "analsex",
        "anilingus",
        "anus",
        "apeshit",
        "ar5e",
        "areola",
        "areole",
        "arian",
        "arrse",
        "arse",
        "arsehole",
        "aryan",
        "ass fuck",
        "ass hole",
        "assbag",
        "assbagger",
        "assbandit",
        "assbang",
        "assbanged",
        "assbanger",
        "assbangs",
        "assbite",
        "assblaster",
        "assclown",
        "asscock",
        "asscracker",
        "asses",
        "assface",
        "assfaces",
        "assfuck",
        "assfucker",
        "ass-fucker",
        "assfukka",
        "assgoblin",
        "assh0le",
        "asshat",
        "ass-hat",
        "asshead",
        "assho1e",
        "asshole",
        "assholes",
        "asshopper",
        "asshore",
        "ass-jabber",
        "assjacker",
        "assjockey",
        "asskiss",
        "asskisser",
        "assklown",
        "asslick",
        "asslicker",
        "asslover",
        "assman",
        "assmaster",
        "assmonkey",
        "assmucus",
        "assmunch",
        "assmuncher",
        "assnigger",
        "asspacker",
        "asspirate",
        "ass-pirate",
        "asspuppies",
        "assranger",
        "assshit",
        "assshole",
        "asssucker",
        "asswad",
        "asswhole",
        "asswhore",
        "asswipe",
        "asswipes",
        "auto erotic",
        "autoerotic",
        "axwound",
        "azazel",
        "azz",
        "b!tch",
        "b00bs",
        "b17ch",
        "b1tch",
        "babeland",
        "badfuck",
        "ball gag",
        "ball gravy",
        "ball kicking",
        "ball licking",
        "ball sack",
        "ball sucking",
        "ballbag",
        "balllicker",
        "ballsack",
        "bampot",
        "bang (one's) box",
        "bangbros",
        "banger",
        "banging",
        "bareback",
        "barely legal",
        "barenaked",
        "barf",
        "barface",
        "barfface",
        "bastard",
        "bastardo",
        "bastards",
        "bastinado",
        "batty boy",
        "bawdy",
        "bazongas",
        "bazooms",
        "bbw",
        "bdsm",
        "beaner",
        "beaners",
        "beardedclam",
        "beastial",
        "beastiality",
        "beatch",
        "beater",
        "beatyourmeat",
        "beaver",
        "beaver cleaver",
        "beaver lips",
        "beef curtain",
        "beef curtains",
        "beeyotch",
        "bellend",
        "bender",
        "beotch",
        "bestial",
        "bestiality",
        "bi+ch",
        "biatch",
        "bicurious",
        "big black",
        "big breasts",
        "big knockers",
        "big tits",
        "bigbastard",
        "bigbutt",
        "bigtits",
        "bimbo",
        "bimbos",
        "bint",
        "birdlock",
        "bisexual",
        "bi-sexual",
        "bitch",
        "bitch tit",
        "bitchass",
        "bitched",
        "bitcher",
        "bitchers",
        "bitches",
        "bitchez",
        "bitchin",
        "bitching",
        "bitchtits",
        "bitchy",
        "black cock",
        "blonde action",
        "blonde on blonde action",
        "bloodclaat",
        "bloody hell",
        "blow job",
        "blow me",
        "blow mud",
        "blow your load",
        "blowjob",
        "blowjobs",
        "blue waffle",
        "blumpkin",
        "boang",
        "bod",
        "bogan",
        "bohunk",
        "boink",
        "boiolas",
        "bollick",
        "bollock",
        "bollocks",
        "bollok",
        "bollox",
        "bomd",
        "boned",
        "boner",
        "boners",
        "bong",
        "boob",
        "boobies",
        "boobs",
        "booby",
        "booger",
        "bookie",
        "boong",
        "boonga",
        "booobs",
        "boooobs",
        "booooobs",
        "booooooobs",
        "bootee",
        "bootie",
        "booty",
        "booty call",
        "booze",
        "boozer",
        "boozy",
        "breastjob",
        "breastlover",
        "breastman",
        "breeder",
        "brotherfucker",
        "brown showers",
        "brunette action",
        "buceta",
        "bugger",
        "buggered",
        "buggery",
        "bukkake",
        "bull shit",
        "bullcrap",
        "bulldike",
        "bulldyke",
        "bullet vibe",
        "bullshit",
        "bullshits",
        "bullshitted",
        "bullturds",
        "bum",
        "bum boy",
        "bumblefuck",
        "bumclat",
        "bumfuck",
        "bummer",
        "bung",
        "bung hole",
        "bunga",
        "bunghole",
        "bunny fucker",
        "bust a load",
        "busty",
        "butchdike",
        "butchdyke",
        "butt",
        "butt fuck",
        "butt plug",
        "buttbang",
        "butt-bang",
        "buttcheeks",
        "buttface",
        "buttfuck",
        "butt-fuck",
        "buttfucka",
        "buttfucker",
        "butt-fucker",
        "butthead",
        "butthole",
        "buttman",
        "buttmuch",
        "buttmunch",
        "buttmuncher",
        "butt-pirate",
        "buttplug",
        "c.0.c.k",
        "c.o.c.k.",
        "c.u.n.t",
        "c0ck",
        "c-0-c-k",
        "c0cksucker",
        "caca",
        "cahone",
        "camel toe",
        "cameltoe",
        "camgirl",
        "camslut",
        "camwhore",
        "carpet muncher",
        "carpetmuncher",
        "cawk",
        "chesticle",
        "chi-chi man",
        "chick with a dick",
        "child-fucker",
        "chinc",
        "chincs",
        "chink",
        "chinky",
        "choad",
        "choade",
        "choc ice",
        "chocolate rosebuds",
        "chode",
        "chodes",
        "chota bags",
        "cipa",
        "circlejerk",
        "cl1t",
        "cleveland steamer",
        "clit",
        "clit licker",
        "clitface",
        "clitfuck",
        "clitoris",
        "clitorus",
        "clits",
        "clitty",
        "clitty litter",
        "clogwog",
        "clover clamps",
        "clunge",
        "clusterfuck",
        "cnut",
        "cocain",
        "cocaine",
        "cock",
        "c-o-c-k",
        "cock pocket",
        "cock snot",
        "cock sucker",
        "cockass",
        "cockbite",
        "cockblock",
        "cockburger",
        "cockeye",
        "cockface",
        "cockfucker",
        "cockhead",
        "cockholster",
        "cockjockey",
        "cockknocker",
        "cockknoker",
        "cocklicker",
        "cocklover",
        "cocklump",
        "cockmaster",
        "cockmongler",
        "cockmongruel",
        "cockmonkey",
        "cockmunch",
        "cockmuncher",
        "cocknose",
        "cocknugget",
        "cocks",
        "cockshit",
        "cocksmith",
        "cocksmoke",
        "cocksmoker",
        "cocksniffer",
        "cocksucer",
        "cocksuck",
        "cocksuck",
        "cocksucked",
        "cocksucker",
        "cock-sucker",
        "cocksuckers",
        "cocksucking",
        "cocksucks",
        "cocksuka",
        "cocksukka",
        "cockwaffle",
        "coffin dodger",
        "coital",
        "cok",
        "cokmuncher",
        "coksucka",
        "commie",
        "condom",
        "coochie",
        "coochy",
        "coon",
        "coonnass",
        "coons",
        "cooter",
        "cop some wood",
        "coprolagnia",
        "coprophilia",
        "corksucker",
        "corp whore",
        "crackwhore",
        "crack-whore",
        "creampie",
        "cretin",
        "crikey",
        "cripple",
        "crotte",
        "cum",
        "cum chugger",
        "cum dumpster",
        "cum freak",
        "cum guzzler",
        "cumbubble",
        "cumdump",
        "cumdumpster",
        "cumguzzler",
        "cumjockey",
        "cummer",
        "cummin",
        "cumming",
        "cums",
        "cumshot",
        "cumshots",
        "cumslut",
        "cumstain",
        "cumtart",
        "cunilingus",
        "cunillingus",
        "cunn",
        "cunnie",
        "cunnilingus",
        "cunntt",
        "cunny",
        "cunt",
        "c-u-n-t",
        "cunt hair",
        "cuntass",
        "cuntbag",
        "cuntface",
        "cuntfuck",
        "cuntfucker",
        "cunthole",
        "cunthunter",
        "cuntlick",
        "cuntlick",
        "cuntlicker",
        "cuntlicker",
        "cuntlicking",
        "cuntrag",
        "cunts",
        "cuntsicle",
        "cuntslut",
        "cunt-struck",
        "cuntsucker",
        "cut rope",
        "cyalis",
        "cyberfuc",
        "cyberfuck",
        "cyberfucked",
        "cyberfucker",
        "cyberfuckers",
        "cyberfucking",
        "cybersex",
        "d0ng",
        "d0uch3",
        "d0uche",
        "d1ck",
        "d1ld0",
        "d1ldo",
        "dago",
        "dagos",
        "dammit",
        "damnit",
        "darkie",
        "darn",
        "date rape",
        "daterape",
        "dawgie-style",
        "deep throat",
        "deepthroat",
        "deggo",
        "dendrophilia",
        "dick head",
        "dick hole",
        "dick shy",
        "dickbag",
        "dickbeaters",
        "dickbrain",
        "dickdipper",
        "dickface",
        "dickflipper",
        "dickfuck",
        "dickfucker",
        "dickhead",
        "dickheads",
        "dickhole",
        "dickish",
        "dick-ish",
        "dickjuice",
        "dickmilk",
        "dickmonger",
        "dickripper",
        "dicksipper",
        "dickslap",
        "dick-sneeze",
        "dicksucker",
        "dicksucking",
        "dicktickler",
        "dickwad",
        "dickweasel",
        "dickweed",
        "dickwhipper",
        "dickwod",
        "dickzipper",
        "diddle",
        "dike",
        "dildo",
        "dildos",
        "diligaf",
        "dillweed",
        "dimwit",
        "dingle",
        "dingleberries",
        "dingleberry",
        "dink",
        "dinks",
        "dipship",
        "dipshit",
        "dirsa",
        "dirty pillows",
        "dirty sanchez",
        "dlck",
        "dog style",
        "dog-fucker",
        "doggie style",
        "doggiestyle",
        "doggie-style",
        "doggin",
        "dogging",
        "doggy style",
        "doggystyle",
        "doggy-style",
        "dolcett",
        "domination",
        "dominatrix",
        "dommes",
        "dong",
        "donkey punch",
        "donkeypunch",
        "donkeyribber",
        "doochbag",
        "doofus",
        "dookie",
        "doosh",
        "dopey",
        "double dong",
        "double penetration",
        "doublelift",
        "douch3",
        "douche",
        "douchebag",
        "douchebags",
        "douche-fag",
        "douchewaffle",
        "douchey",
        "dp action",
        "dry hump",
        "duche",
        "dumass",
        "dumb ass",
        "dumbass",
        "dumbasses",
        "dumbcunt",
        "dumbfuck",
        "dumbshit",
        "dummy",
        "dumshit",
        "dvda",
        "dyke",
        "dykes",
        "eat a dick",
        "eat hair pie",
        "eat my ass",
        "eatpussy",
        "ecchi",
        "ejaculate",
        "ejaculated",
        "ejaculates",
        "ejaculating",
        "ejaculatings",
        "ejaculation",
        "ejakulate",
        "erection",
        "erotic",
        "erotism",
        "essohbee",
        "eunuch",
        "extacy",
        "extasy",
        "f u c k",
        "f u c k e r",
        "f.u.c.k",
        "f_u_c_k",
        "f4nny",
        "facefucker",
        "facial",
        "fack",
        "fag",
        "fagbag",
        "fagfucker",
        "fagg",
        "fagged",
        "fagging",
        "faggit",
        "faggitt",
        "faggot",
        "faggotcock",
        "faggots",
        "faggs",
        "fagot",
        "fagots",
        "fags",
        "fagtard",
        "faig",
        "faigt",
        "fanny",
        "fannybandit",
        "fannyflaps",
        "fannyfucker",
        "fanyy",
        "fastfuck",
        "fatass",
        "fatfuck",
        "fatfucker",
        "fcuk",
        "fcuker",
        "fcuking",
        "fecal",
        "feck",
        "fecker",
        "felch",
        "felcher",
        "felching",
        "fellate",
        "fellatio",
        "feltch",
        "feltcher",
        "female squirting",
        "femdom",
        "fenian",
        "figging",
        "fingerbang",
        "fingerfuck",
        "fingerfuck",
        "fingerfucked",
        "fingerfucker",
        "fingerfucker",
        "fingerfuckers",
        "fingerfucking",
        "fingerfucks",
        "fingering",
        "fist fuck",
        "fisted",
        "fistfuck",
        "fistfucked",
        "fistfucker",
        "fistfucker",
        "fistfuckers",
        "fistfucking",
        "fistfuckings",
        "fistfucks",
        "fisting",
        "fisty",
        "flamer",
        "flange",
        "flaps",
        "fleshflute",
        "flog the log",
        "floozy",
        "foad",
        "foah",
        "foobar",
        "fook",
        "fooker",
        "foot fetish",
        "footfuck",
        "footfucker",
        "footjob",
        "footlicker",
        "freakfuck",
        "freakyfucker",
        "freefuck",
        "freex",
        "frigg",
        "frigga",
        "frotting",
        "fubar",
        "fuc",
        "fuck",
        "f-u-c-k",
        "fuck buttons",
        "fuck hole",
        "fuck off",
        "fuck puppet",
        "fuck trophy",
        "fuck yo mama",
        "fuck you",
        "fucka",
        "fuckass",
        "fuck-ass",
        "fuckbag",
        "fuck-bitch",
        "fuckboy",
        "fuckbrain",
        "fuckbutt",
        "fuckbutter",
        "fucked",
        "fuckedup",
        "fucker",
        "fuckers",
        "fuckersucker",
        "fuckface",
        "fuckfreak",
        "fuckhead",
        "fuckheads",
        "fuckher",
        "fuckhole",
        "fuckin",
        "fucking",
        "fuckingbitch",
        "fuckings",
        "fuckingshitmotherfucker",
        "fuckme",
        "fuckme",
        "fuckmeat",
        "fuckmehard",
        "fuckmonkey",
        "fucknugget",
        "fucknut",
        "fucknutt",
        "fuckoff",
        "fucks",
        "fuckstick",
        "fucktard",
        "fuck-tard",
        "fucktards",
        "fucktart",
        "fucktoy",
        "fucktwat",
        "fuckup",
        "fuckwad",
        "fuckwhit",
        "fuckwhore",
        "fuckwit",
        "fuckwitt",
        "fuckyou",
        "fudge packer",
        "fudgepacker",
        "fudge-packer",
        "fuk",
        "fuker",
        "fukker",
        "fukkers",
        "fukkin",
        "fuks",
        "fukwhit",
        "fukwit",
        "fuq",
        "futanari",
        "fux",
        "fux0r",
        "fvck",
        "fxck",
        "gae",
        "gai",
        "gang bang",
        "gangbang",
        "gang-bang",
        "gangbanged",
        "gangbangs",
        "ganja",
        "gash",
        "gassy ass",
        "gay sex",
        "gayass",
        "gaybob",
        "gaydo",
        "gayfuck",
        "gayfuckist",
        "gaylord",
        "gaysex",
        "gaytard",
        "gaywad",
        "gender bender",
        "genitals",
        "gey",
        "gfy",
        "ghay",
        "ghey",
        "giant cock",
        "gigolo",
        "ginger",
        "gippo",
        "girl on",
        "girl on top",
        "girls gone wild",
        "glans",
        "goatcx",
        "goatse",
        "god damn",
        "godamn",
        "godamnit",
        "goddam",
        "god-dam",
        "goddammit",
        "goddamn",
        "goddamned",
        "god-damned",
        "goddamnit",
        "goddamnmuthafucker",
        "godsdamn",
        "gokkun",
        "golden shower",
        "goldenshower",
        "golliwog",
        "gonad",
        "gonads",
        "gonorrehea",
        "goo girl",
        "gooch",
        "goodpoop",
        "gook",
        "gooks",
        "goregasm",
        "gotohell",
        "gringo",
        "grope",
        "group sex",
        "gspot",
        "g-spot",
        "gtfo",
        "guido",
        "guro",
        "h0m0",
        "h0mo",
        "ham flap",
        "hand job",
        "handjob",
        "hard core",
        "hard on",
        "hardcore",
        "hardcoresex",
        "he11",
        "headfuck",
        "hebe",
        "heeb",
        "hell",
        "hentai",
        "heroin",
        "herp",
        "herpes",
        "herpy",
        "heshe",
        "he-she",
        "hitler",
        "hiv",
        "ho",
        "hoar",
        "hoare",
        "hobag",
        "hoe",
        "hoer",
        "holy shit",
        "hom0",
        "homey",
        "homo",
        "homodumbshit",
        "homoerotic",
        "homoey",
        "honkey",
        "honky",
        "hooch",
        "hookah",
        "hooker",
        "hoor",
        "hootch",
        "hooter",
        "hore",
        "horniest",
        "horny",
        "hot carl",
        "hot chick",
        "hotpussy",
        "hotsex",
        "how to kill",
        "how to murdep",
        "how to murder",
        "huge fat",
        "humped",
        "humping",
        "hun",
        "hussy",
        "hymen",
        "iap",
        "iberian slap",
        "inbred",
        "incest",
        "injun",
        "intercourse",
        "j3rk0ff",
        "jack off",
        "jackass",
        "jackasses",
        "jackhole",
        "jackoff",
        "jack-off",
        "jaggi",
        "jagoff",
        "jail bait",
        "jailbait",
        "jap",
        "japs",
        "jerk off",
        "jerk0ff",
        "jerkass",
        "jerked",
        "jerkoff",
        "jerk-off",
        "jigaboo",
        "jiggaboo",
        "jiggerboo",
        "jism",
        "jiz",
        "jizm",
        "jizz",
        "jizzed",
        "jock",
        "juggs",
        "jungle bunny",
        "junglebunny",
        "junkie",
        "junky",
        "kafir",
        "kawk",
        "kike",
        "kikes",
        "kinbaku",
        "kinkster",
        "kinky",
        "kkk",
        "klan",
        "knob",
        "knob end",
        "knobbing",
        "knobead",
        "knobed",
        "knobend",
        "knobhead",
        "knobjocky",
        "knobjokey",
        "kock",
        "kondum",
        "kondums",
        "kooch",
        "kooches",
        "kootch",
        "kraut",
        "kum",
        "kummer",
        "kumming",
        "kums",
        "kunilingus",
        "kunja",
        "kunt",
        "kwif",
        "kyke",
        "l3i+ch",
        "l3itch",
        "labia",
        "lameass",
        "lardass",
        "leather restraint",
        "leather straight jacket",
        "lech",
        "lemon party",
        "leper",
        "lesbo",
        "lesbos",
        "lez",
        "lezbian",
        "lezbians",
        "lezbo",
        "lezbos",
        "lezza",
        "lezzie",
        "lezzies",
        "lezzy",
        "lolita",
        "lovemaking",
        "lust",
        "lusting",
        "lusty",
        "m0f0",
        "m0fo",
        "m45terbate",
        "ma5terb8",
        "ma5terbate",
        "mafugly",
        "make me come",
        "male squirting",
        "mams",
        "masochist",
        "massa",
        "masterb8",
        "masterbat",
        "masterbat3",
        "masterbate",
        "master-bate",
        "masterbating",
        "masterbation",
        "masterbations",
        "masturbate",
        "masturbating",
        "masturbation",
        "maxi",
        "mcfagget",
        "menage a trois",
        "menses",
        "meth",
        "m-fucking",
        "mick",
        "middle finger",
        "midget",
        "milf",
        "minge",
        "minger",
        "missionary position",
        "mof0",
        "mofo",
        "mo-fo",
        "molest",
        "mong",
        "moo moo foo foo",
        "moolie",
        "moron",
        "mothafuck",
        "mothafucka",
        "mothafuckas",
        "mothafuckaz",
        "mothafucked",
        "mothafucker",
        "mothafuckers",
        "mothafuckin",
        "mothafucking",
        "mothafuckings",
        "mothafucks",
        "mother fucker",
        "motherfuck",
        "motherfucka",
        "motherfucked",
        "motherfucker",
        "motherfuckers",
        "motherfuckin",
        "motherfucking",
        "motherfuckings",
        "motherfuckka",
        "motherfucks",
        "mound of venus",
        "mr hands",
        "mtherfucker",
        "mthrfucker",
        "mthrfucking",
        "muff",
        "muff diver",
        "muff puff",
        "muffdiver",
        "muffdiving",
        "munging",
        "munter",
        "murder",
        "mutha",
        "muthafecker",
        "muthafuckaz",
        "muthafuckker",
        "muther",
        "mutherfucker",
        "mutherfucking",
        "muthrfucking",
        "n1gga",
        "n1gger",
        "nad",
        "nads",
        "naked",
        "nambla",
        "napalm",
        "nappy",
        "nawashi",
        "nazi",
        "nazism",
        "need the dick",
        "negro",
        "neonazi",
        "nig nog",
        "nigaboo",
        "nigg3r",
        "nigg4h",
        "nigga",
        "niggah",
        "niggas",
        "niggaz",
        "nigger",
        "niggers",
        "niggle",
        "niglet",
        "nig-nog",
        "nimphomania",
        "nimrod",
        "ninny",
        "nob",
        "nob jokey",
        "nobhead",
        "nobjocky",
        "nobjokey",
        "nonce",
        "nooky",
        "nsfw images",
        "numbnuts",
        "nut butter",
        "nut sack",
        "nutsack",
        "nutter",
        "nympho",
        "nymphomania",
        "octopussy",
        "old bag",
        "omorashi",
        "one cup two girls",
        "one guy one jar",
        "opiate",
        "opium",
        "orgasim",
        "orgasims",
        "orgasm",
        "orgasmic",
        "orgasms",
        "orgies",
        "orgy",
        "ovum",
        "ovums",
        "p.u.s.s.y.",
        "p0rn",
        "paddy",
        "paedophile",
        "paki",
        "panooch",
        "pansy",
        "pcp",
        "pecker",
        "peckerhead",
        "pedo",
        "pedobear",
        "pedophile",
        "pedophilia",
        "pedophiliac",
        "peepee",
        "pegging",
        "penetrate",
        "penetration",
        "penial",
        "penile",
        "penis",
        "penisbanger",
        "penisfucker",
        "penispuffer",
        "perversion",
        "peyote",
        "phalli",
        "phallic",
        "phone sex",
        "phonesex",
        "phuck",
        "phuk",
        "phuked",
        "phuking",
        "phukked",
        "phukking",
        "phuks",
        "phuq",
        "piece of shit",
        "pigfucker",
        "pikey",
        "pillowbiter",
        "pimp",
        "pimpis",
        "pinko",
        "piss",
        "piss pig",
        "pissed",
        "pisser",
        "pissers",
        "pisses",
        "pissflaps",
        "pissin",
        "pissing",
        "pissoff",
        "piss-off",
        "pisspig",
        "playboy",
        "pleasure chest",
        "polack",
        "pole smoker",
        "polesmoker",
        "pollock",
        "ponyplay",
        "poof",
        "poon",
        "poonani",
        "poonany",
        "poontang",
        "poop",
        "poop chute",
        "poopchute",
        "poopuncher",
        "porch monkey",
        "porchmonkey",
        "porn",
        "porno",
        "pornography",
        "pornos",
        "potty",
        "prick",
        "pricks",
        "prickteaser",
        "prig",
        "prince albert piercing",
        "prod",
        "pron",
        "prostitute",
        "prude",
        "psycho",
        "pthc",
        "pube",
        "pubes",
        "pubic",
        "pubis",
        "punani",
        "punanny",
        "punany",
        "punkass",
        "punky",
        "punta",
        "puss",
        "pusse",
        "pussi",
        "pussies",
        "pussy",
        "pussy fart",
        "pussy palace",
        "pussylicking",
        "pussypounder",
        "pussys",
        "pust",
        "puto",
        "queaf",
        "queef",
        "queerbait",
        "queerhole",
        "queero",
        "queers",
        "quicky",
        "quim",
        "racy",
        "raghead",
        "raging boner",
        "rape",
        "raped",
        "raper",
        "rapey",
        "raping",
        "rapist",
        "raunch",
        "rectus",
        "reefer",
        "reetard",
        "reich",
        "renob",
        "retard",
        "retarded",
        "reverse cowgirl",
        "revue",
        "rimjaw",
        "rimjob",
        "rimming",
        "ritard",
        "rosy palm",
        "rosy palm and her 5 sisters",
        "rtard",
        "r-tard",
        "rubbish",
        "rum",
        "rump",
        "rumprammer",
        "ruski",
        "rusty trombone",
        "s hit",
        "s&m",
        "s.h.i.t.",
        "s.o.b.",
        "s_h_i_t",
        "s0b",
        "sadism",
        "sadist",
        "sambo",
        "sand nigger",
        "sandler",
        "sandnigger",
        "sanger",
        "sausage queen",
        "scag",
        "scantily",
        "scat",
        "schizo",
        "schlong",
        "scissoring",
        "screwed",
        "screwing",
        "scroat",
        "scrog",
        "scrot",
        "scrote",
        "scrotum",
        "scrud",
        "scum",
        "seaman",
        "seamen",
        "seduce",
        "seks",
        "semen",
        "sex",
        "sexo",
        "sexual",
        "sexy",
        "sh!+",
        "sh!t",
        "sh1t",
        "s-h-1-t",
        "shag",
        "shagger",
        "shaggin",
        "shagging",
        "shamedame",
        "shaved beaver",
        "shaved pussy",
        "shemale",
        "shi+",
        "shibari",
        "shirt lifter",
        "shit",
        "s-h-i-t",
        "shit ass",
        "shit fucker",
        "shitass",
        "shitbag",
        "shitbagger",
        "shitblimp",
        "shitbrains",
        "shitbreath",
        "shitcanned",
        "shitcunt",
        "shitdick",
        "shite",
        "shiteater",
        "shited",
        "shitey",
        "shitface",
        "shitfaced",
        "shitfuck",
        "shitfull",
        "shithead",
        "shitheads",
        "shithole",
        "shithouse",
        "shiting",
        "shitings",
        "shits",
        "shitspitter",
        "shitstain",
        "shitt",
        "shitted",
        "shitter",
        "shitters",
        "shittier",
        "shittiest",
        "shitting",
        "shittings",
        "shitty",
        "shiz",
        "shiznit",
        "shota",
        "shrimping",
        "sissy",
        "skag",
        "skank",
        "skeet",
        "skullfuck",
        "slag",
        "slanteye",
        "slave",
        "sleaze",
        "sleazy",
        "slut",
        "slut bucket",
        "slutbag",
        "slutdumper",
        "slutkiss",
        "sluts",
        "smartass",
        "smartasses",
        "smeg",
        "smegma",
        "smut",
        "smutty",
        "snowballing",
        "snuff",
        "s-o-b",
        "sod off",
        "sodom",
        "sodomize",
        "sodomy",
        "son of a bitch",
        "son of a motherless goat",
        "son of a whore",
        "son-of-a-bitch",
        "souse",
        "soused",
        "spac",
        "spic",
        "spick",
        "spik",
        "spiks",
        "splooge",
        "splooge moose",
        "spooge",
        "spook",
        "spread legs",
        "spunk",
        "stfu",
        "stiffy",
        "stoned",
        "strap on",
        "strapon",
        "strappado",
        "strip club",
        "style doggy",
        "suckass",
        "suicide girls",
        "sultry women",
        "sumofabiatch",
        "swastika",
        "swinger",
        "t1t",
        "t1tt1e5",
        "t1tties",
        "taff",
        "taig",
        "tainted love",
        "taking the piss",
        "tard",
        "tawdry",
        "tea bagging",
        "teabagging",
        "teat",
        "teets",
        "teez",
        "terd",
        "teste",
        "testee",
        "testes",
        "testical",
        "testicle",
        "testis",
        "threesome",
        "throating",
        "thug",
        "thundercunt",
        "tied up",
        "tight white",
        "tinkle",
        "tit",
        "tit wank",
        "titfuck",
        "titi",
        "tities",
        "tits",
        "titt",
        "tittie5",
        "tittiefucker",
        "titties",
        "titty",
        "tittyfuck",
        "tittyfucker",
        "tittywank",
        "titwank",
        "toke",
        "tongue in a",
        "toots",
        "tosser",
        "towelhead",
        "tranny",
        "tribadism",
        "tub girl",
        "tubgirl",
        "turd",
        "tush",
        "tushy",
        "tw4t",
        "twat",
        "twathead",
        "twatlips",
        "twats",
        "twatty",
        "twatwaffle",
        "twink",
        "two fingers",
        "two fingers with tongue",
        "two girls one cup",
        "twunt",
        "twunter",
        "unclefucker",
        "undressing",
        "unwed",
        "upskirt",
        "urethra play",
        "urine",
        "urophilia",
        "uzi",
        "v14gra",
        "v1gra",
        "vag",
        "vagina",
        "vajayjay",
        "va-j-j",
        "valium",
        "venus mound",
        "veqtable",
        "violet wand",
        "virgin",
        "vixen",
        "vjayjay",
        "vorarephilia",
        "voyeur",
        "vulgar",
        "vulva",
        "w00se",
        "wad",
        "wang",
        "wank",
        "wanker",
        "wankjob",
        "wanky",
        "wazoo",
        "wench",
        "wet dream",
        "wetback",
        "wh0re",
        "wh0reface",
        "white power",
        "whitey",
        "whoar",
        "whoralicious",
        "whore",
        "whorealicious",
        "whorebag",
        "whored",
        "whoreface",
        "whorehopper",
        "whorehouse",
        "whores",
        "whoring",
        "wigger",
        "window licker",
        "wiseass",
        "wiseasses",
        "wog",
        "wop",
        "wrapping men",
        "wrinkled starfish",
        "wtf",
        "xrated",
        "x-rated",
        "xx",
        "xxx",
        "yaoi",
        "yeasty",
        "yellow showers",
        "yid",
        "yiffy",
        "yobbo",
        "zoophile",
        "zoophilia",
        "zubb",
        "abo",
        "abbo",
        "gator bait",
        "alligator bait",
        "alpine serb",
        "ah chah",
        "ang mo",
        "ape",
        "beaner",
        "beaney",
        "bluegum",
        "bog",
        "bogtrotter",
        "bog-trotter",
        "bohunk",
        "bootlip",
        "buddhahead",
        "burrhead",
        "burr-head",
        "burr head",
        "cabbage eater",
        "camel jockey",
        "chinaman",
        "ching chong",
        "chink",
        "christ-killer",
        "choc-ice",
        "cholo",
        "chug",
        "coolie",
        "coon",
        "coonass",
        "coon-ass",
        "darky",
        "darkey",
        "drakie",
        "dune coon",
        "gook",
        "gooky",
        "goombah",
        "goy",
        "goyim",
        "goyum",
        "greaseball",
        "greaser",
        "gringo",
        "groid",
        "gub",
        "gubba",
        "gwer",
        "half-breed",
        "half-caste",
        "hillbilly",
        "injun",
        "jap",
        "jewboy",
        "jigaboo",
        "jiggabo",
        "jigarooni",
        "jijjiboo",
        "zigabo",
        "jig",
        "jigg",
        "jigger",
        "jungle bunny",
        "mayonnaise monkey",
        "mau-mau",
        "northern monkey",
        "dirty monkey",
        "oven dodger",
        "paleface",
        "pancake face",
        "papoose",
        "pickaninny",
        "plastic paddy",
        "prairie nigger",
        "quashie",
        "raghead",
        "rastus",
        "redlegs",
        "redskin",
        "roundeye",
        "russki",
        "ruski",
        "sand nigger",
        "sheepshagger",
        "cooter",
        "slant",
        "slant-eye",
        "slopehead",
        "sloper",
        "sooty",
        "spic",
        "spig",
        "squarehead",
        "squaw",
        "tacohead",
        "ting tong",
        "towel head",
        "wagon burner",
        "white interloper",
        "black interloper",
        "white trash",
        "black trash",
        "whitey",
        "wog",
        "wop",
        "yam yam",
        "yid",
        "zipperhead",
        "baby batter",
        "imbecile",
        "midget",
        "retarded",
    ]
)
