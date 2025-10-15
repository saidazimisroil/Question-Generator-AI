"""Generate a JSONL dataset of Uzbek question and answer pairs."""
from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


QUESTION_TEMPLATES = [
    "{name} haqida qisqacha ma'lumot bera olasizmi?",
    "{name} nimasi bilan mashhur?",
    "{name} qanday ahamiyatga ega?",
    "{name} qayerda yoki qaysi sohada uchraydi?",
    "{name}ning tarixiy o'rni qanday?",
    "{name} bugungi kunda qanday rol o'ynaydi?",
    "{name} bilan bog'liq eng muhim fakt nima?",
    "{name} qaysi madaniy yoki ilmiy an'analarga taalluqlidir?",
    "{name} nega o'zbeklar uchun muhim hisoblanadi?",
    "{name}ning asosiy xususiyati nimada?",
]

FACTS = [
    {
        "name": "Toshkent",
        "answer": "Toshkent O'zbekistonning poytaxti va eng yirik shaharidir. Shahar Markaziy Osiyodagi muhim madaniy va iqtisodiy markaz sifatida tanilgan.",
    },
    {
        "name": "Samarqand",
        "answer": "Samarqand qadimiy Ipak yo'li bo'ylab joylashgan tarixiy shahar bo'lib, ko'plab me'moriy obidalari bilan mashhur. Registon majmuasi va Ulug'bek rasadxonasi bu yerning eng yorqin ramzlaridan hisoblanadi.",
    },
    {
        "name": "Buxoro",
        "answer": "Buxoro O'rta Osiyoning eng qadimiy shaharlaridan biri bo'lib, islomiy ilmlar markazi sifatida tarixda chuqur iz qoldirgan. Uning madrasalari va masjidlari o'rta asr me'morchiligining durdonalaridir.",
    },
    {
        "name": "Xiva",
        "answer": "Xiva qadimiy Xorazm davlatining markazlaridan biri bo'lib, Ichan-Qal'a qo'rg'oni bilan mashhur. Shahar UNESCOning Butunjahon merosi ro'yxatiga kiritilgan.",
    },
    {
        "name": "Farg'ona vodiysi",
        "answer": "Farg'ona vodiysi serhosil tuproqlari va boy dehqonchilik an'analari bilan tanilgan hududdir. Vodiyning tumanlari o'ziga xos hunarmandchiligi va folklori bilan ajralib turadi.",
    },
    {
        "name": "Nukus",
        "answer": "Nukus Qoraqalpog'iston Respublikasining markazi va Amudaryo bo'yida joylashgan shahar. U Savitskiy nomidagi San'at muzeyi bilan dunyoga mashhur.",
    },
    {
        "name": "Termiz",
        "answer": "Termiz Surxondaryo viloyatining markazi bo'lib, qadimiy Budda yodgorliklari va arxeologik topilmalari bilan tanilgan. Shahar Afg'oniston bilan chegaradosh strategik nuqta hisoblanadi.",
    },
    {
        "name": "Andijon",
        "answer": "Andijon Farg'ona vodiysidagi yirik sanoat va madaniy markazdir. Shahar mashhur adib Zahiriddin Muhammad Boburning vatani sifatida ham e'tirof etiladi.",
    },
    {
        "name": "Namangan",
        "answer": "Namangan o'zining bog'lari, gulchilik an'analari va hunarmandchiligi bilan tanilgan shahar. U Farg'ona vodiysidagi yirik ma'naviy markazlardan biridir.",
    },
    {
        "name": "Qarshi",
        "answer": "Qarshi Qashqadaryo viloyatining ma'muriy markazi bo'lib, paxtachilik va chorvachilik rivojlangan hududdir. Shahar yaqinida qadimiy Kitob rasadxonasi joylashgan.",
    },
    {
        "name": "Jizzax",
        "answer": "Jizzax Zarafshon va Sirdaryo oralig'idagi strategik darvoza sifatida tarixda ahamiyat kasb etgan. Bugungi kunda u transport va qishloq xo'jaligi markazi hisoblanadi.",
    },
    {
        "name": "Guliston",
        "answer": "Guliston Sirdaryo viloyatining markazi bo'lib, paxta va g'alla yetishtirish bilan mashhur. Shahar hududida ko'plab irrigatsiya kanallari mavjud.",
    },
    {
        "name": "Urganch",
        "answer": "Urganch Xorazm viloyatining markazi va Amudaryo bo'yidagi muhim transport tugunidir. Shahar Xiva va boshqa tarixiy joylarga kirish eshigi sifatida xizmat qiladi.",
    },
    {
        "name": "Qo'qon",
        "answer": "Qo'qon XVIII-XIX asrlarda Qo'qon xonligining poytaxti bo'lgan tarixiy shahardir. Shahar me'morchiligida madrasalar va saroylar alohida o'rin tutadi.",
    },
    {
        "name": "Shahrisabz",
        "answer": "Shahrisabz Amir Temurning tug'ilgan shahri sifatida mashhur. Bu yerda Oqsaroy saroyi va Jahongir maqbarasi kabi yodgorliklar saqlanib qolgan.",
    },
    {
        "name": "Navoiy",
        "answer": "Navoiy cho'l hududini o'zlashtirish natijasida barpo etilgan sanoat shahri. U metallurgiya va kimyo sanoati bilan ajralib turadi.",
    },
    {
        "name": "Qoraqalpog'iston",
        "answer": "Qoraqalpog'iston O'zbekiston tarkibidagi muxtor respublika bo'lib, o'ziga xos til va madaniyatga ega. Hudud Aral dengizi inqirobi bilan bog'liq ekologik muammolarni boshdan kechirmoqda.",
    },
    {
        "name": "Aral dengizi",
        "answer": "Aral dengizi bir vaqtlar dunyodagi yirik ko'llardan biri bo'lgan, ammo sug'orish siyosati natijasida keskin qurib qolgan. Bugungi kunda u ekologik fojia ramziga aylangan.",
    },
    {
        "name": "Zarafshon daryosi",
        "answer": "Zarafshon daryosi Samarqand va Buxoro viloyatlari orqali oqib o'tuvchi suv manbai hisoblanadi. Daryo vohasida qadimdan dehqonchilik va bog'dorchilik rivojlangan.",
    },
    {
        "name": "Amudaryo",
        "answer": "Amudaryo Markaziy Osiyodagi eng katta daryolardan biri bo'lib, Orol havzasini suv bilan ta'minlab kelgan. Daryo bo'yidagi vohalar qadimdan tsivilizatsiya markazlari bo'lib kelgan.",
    },
    {
        "name": "Sirdaryo",
        "answer": "Sirdaryo O'zbekiston shimoli va sharqini kesib o'tuvchi muhim daryodir. Uning suv resurslari energetika va sug'orish tizimlari uchun katta ahamiyatga ega.",
    },
    {
        "name": "Chotqol tog'lari",
        "answer": "Chotqol tog'lari Tyan-Shan tizimining g'arbiy qismida joylashgan va boy flora hamda faunaga ega. Hudud ekoturizm va alpinistlar uchun mashhur maskan hisoblanadi.",
    },
    {
        "name": "Tyan-Shan",
        "answer": "Tyan-Shan tog' tizimi Markaziy Osiyoni qamrab olgan ulkan tabiiy to'siqdir. U ko'plab muzliklar va daryolarning manbai sifatida ahamiyatlidir.",
    },
    {
        "name": "Hisor tog'lari",
        "answer": "Hisor tog'lari O'zbekistonning janubida joylashgan bo'lib, baland cho'qqilari bilan ajralib turadi. Bu hududda ko'plab milliy tabiat bog'lari tashkil etilgan.",
    },
    {
        "name": "Orolbo'yi",
        "answer": "Orolbo'yi Aral dengizi atrofi hududlarini qamrab oladi va ekologik tiklanish loyihalarining markaziga aylangan. Bu yerda yangi o'rmonzorlar barpo etish tashabbuslari olib borilmoqda.",
    },
    {
        "name": "Mirzo Ulug'bek",
        "answer": "Mirzo Ulug'bek Temuriylar sulolasining ilm-fanga berilgan hukmdori bo'lib, astronomik kuzatuvlari bilan mashhur. Uning Samarqanddagi rasadxonasi yulduzlar jadvalini yaratishda muhim rol o'ynagan.",
    },
    {
        "name": "Alisher Navoiy",
        "answer": "Alisher Navoiy o'zbek adabiyotining asoschisi va buyuk mutafakkir hisoblanadi. Uning «Xamsa» asari turkiy adabiyotda yangi badiiy ufqlarni ochgan.",
    },
    {
        "name": "Amir Temur",
        "answer": "Amir Temur XIV asrda kuchli imperiya barpo etgan sarkarda va davlat arbobidir. Uning siyosati markazlashgan boshqaruv va savdo yo'llarining himoyasiga qaratilgan edi.",
    },
    {
        "name": "Zahiriddin Muhammad Bobur",
        "answer": "Zahiriddin Muhammad Bobur hindistonlik Boburiylar sulolasining asoschisi va iste'dodli adib hisoblanadi. U «Boburnoma» asarida o'z hayoti va davr voqealarini batafsil tasvirlagan.",
    },
    {
        "name": "Muhammad Al-Xorazmiy",
        "answer": "Muhammad Al-Xorazmiy algebra fanining asoschilaridan biri sifatida tanilgan. Uning ilmiy asarlari matematika va astronomiyada inqilobiy yutuqlarni olib kelgan.",
    },
    {
        "name": "Abu Rayhon Beruniy",
        "answer": "Abu Rayhon Beruniy ko'plab ilmiy sohalarda tadqiqot olib borgan ensiklopedist olimdir. U geodeziya, tarix va dorivor o'simliklar haqida bebaho ma'lumotlar qoldirgan.",
    },
    {
        "name": "Abu Ali ibn Sino",
        "answer": "Abu Ali ibn Sino tibbiyot va falsafa sohalaridagi asarlari bilan butun dunyoga mashhur bo'lgan allomadir. Uning «Tib qonunlari» asari asrlar davomida asosiy qo'llanma bo'lib xizmat qilgan.",
    },
    {
        "name": "Mahmud Qoshg'ariy",
        "answer": "Mahmud Qoshg'ariy turkiy tillarni o'rgangan va «Devonu lug'otit turk» asarini yaratgan tilshunosdir. U turkiy xalqlarning tarixi va madaniyatini tizimli ravishda bayon qilgan.",
    },
    {
        "name": "Ogahiy",
        "answer": "Ogahiy XIX asr Xorazm adabiyotining yirik vakili bo'lib, tarixiy va lirik asarlar yaratgan. Uning ijodi xonlik hayotini keng yoritadi.",
    },
    {
        "name": "G'iyosiddin Jamshid Koshiy",
        "answer": "G'iyosiddin Jamshid Koshiy matematika va astronomiya sohasidagi aniqliklari bilan mashhur bo'lgan olim. U pi sonining aniq qiymatini hisoblashda katta natijalarga erishgan.",
    },
    {
        "name": "Nodira begim",
        "answer": "Nodira begim Qo'qon xonligi saroyidagi ma'rifatparvar shoir va davlat arbobi bo'lgan. U she'riyatda ayollar ovozini yuksaltirishga katta hissa qo'shgan.",
    },
    {
        "name": "Berdaq",
        "answer": "Berdaq qoraqalpoq adabiyotining asoschilaridan biri hisoblanadi. Uning doston va qo'shiqlari xalqning dardini va orzularini ifoda etadi.",
    },
    {
        "name": "Hamza Hakimzoda Niyoziy",
        "answer": "Hamza Hakimzoda Niyoziy jadidchilik harakatining faol namoyandasi va teatr asoschilaridan biridir. U ma'rifatli jamiyat qurish g'oyalarini targ'ib etgan.",
    },
    {
        "name": "Abdulla Qodiriy",
        "answer": "Abdulla Qodiriy o'zbek romanchiligining asoschisi bo'lib, «O'tkan kunlar» asari bilan tanilgan. Uning ijodi milliy ongni uyg'otishda muhim rol o'ynagan.",
    },
    {
        "name": "Cho'lpon",
        "answer": "Cho'lpon jadid adabiyoti vakili bo'lib, ozodlik va erkinlik g'oyalarini targ'ib qilgan shoirdir. Uning «Kecha va kunduz» romani o'z davrining keskin masalalarini yoritadi.",
    },
    {
        "name": "G'afur G'ulom",
        "answer": "G'afur G'ulom o'zbek sovet adabiyotining yorqin vakili bo'lib, satirik va dramatik asarlar yaratgan. Uning «Shum bola» qissasi xalq orasida juda mashhur.",
    },
    {
        "name": "Abdulla Oripov",
        "answer": "Abdulla Oripov O'zbekiston xalq shoiri va davlat madhiyasi muallifidir. Uning she'rlari vatanparvarlik va ma'naviy qadriyatlarni ulug'laydi.",
    },
    {
        "name": "Erkin Vohidov",
        "answer": "Erkin Vohidov she'riyatda lirika va falsafiy g'oyalarni uyg'unlashtirgan ijodkor hisoblanadi. Uning asarlari milliy o'zlikni anglashga chorlaydi.",
    },
    {
        "name": "Said Ahmad",
        "answer": "Said Ahmad o'zbek dramaturgiyasi va prozasiga sezilarli ta'sir ko'rsatgan yozuvchidir. Uning sahna asarlari jamiyatdagi ijtimoiy muammolarni yoritadi.",
    },
    {
        "name": "Zulfiya",
        "answer": "Zulfiya o'zbek ayollar she'riyatining ramzi bo'lib, muhabbat va sadoqat mavzularini tarannum etgan. U ko'plab xalqaro adabiy mukofotlarga sazovor bo'lgan.",
    },
    {
        "name": "Navro'z bayrami",
        "answer": "Navro'z bahor va yangi yilni qarshi olish bilan bog'liq qadimiy xalq bayramidir. U tabiatning uyg'onishi va ezgulik g'oyalarini mujassam etadi.",
    },
    {
        "name": "Sumalak",
        "answer": "Sumalak Navro'z bayramida tayyorlanadigan an'anaviy shirin taomdir. Uni tayyorlash jarayoni hamjihatlik va mehr-oqibat ramziga aylangan.",
    },
    {
        "name": "Palov",
        "answer": "Palov o'zbek milliy oshxonasi ramzi bo'lib, guruch, go'sht va sabzavotlardan tayyorlanadi. U mehmondo'stlik va to'y-ma'rakalarning ajralmas qismidir.",
    },
    {
        "name": "Atlas matosi",
        "answer": "Atlas ipakdan to'qiladigan rang-barang naqshli matodir. U Farg'ona vodiysida qadimdan buyon ayollar liboslarini bezab keladi.",
    },
    {
        "name": "O'zbek qovuni",
        "answer": "O'zbek qovuni shirin ta'mi va xilma-xilligi bilan mashhur bo'lgan bog'dorchilik mahsulidir. Mamlakatda qovun yetishtirish bo'yicha boy tajriba mavjud.",
    },
    {
        "name": "Hazrati Imom majmuasi",
        "answer": "Hazrati Imom majmuasi Toshkentdagi diniy-ma'rifiy markaz bo'lib, qadimiy qo'lyozmalarni saqlaydi. Bu yerda mashhur Usmon Qur'onining nusxasi namoyish etiladi.",
    },
    {
        "name": "Ichan-Qal'a",
        "answer": "Ichan-Qal'a Xiva shahridagi ichki qal'a bo'lib, ko'plab madrasalar va masjidlarni o'z ichiga oladi. Majmua ko'hna Xorazm me'morchiligining yorqin namunasi hisoblanadi.",
    },
    {
        "name": "Ark qal'asi",
        "answer": "Ark qal'asi Buxoro amirlarining saroyi bo'lib, siyosiy va ma'muriy markaz vazifasini o'tagan. Qal'a devorlari ichida ko'plab tarixiy binolar joylashgan.",
    },
    {
        "name": "Minorai Kalon",
        "answer": "Minorai Kalon Buxorodagi mashhur minora bo'lib, XII asrda qurilgan. U shaharning osmonida o'ziga xos siluet yaratadi.",
    },
    {
        "name": "Sherdor madrasasi",
        "answer": "Sherdor madrasasi Samarqanddagi Registon majmuasining markaziy inshootlaridan biridir. Uning fasadidagi sher tasvirlari bilan tanilgan.",
    },
    {
        "name": "Tilla-Kori madrasasi",
        "answer": "Tilla-Kori madrasasi oltin bilan bezatilgan gumbaz va interyeri bilan mashhur. Bu madrasa Registon majmuasining muhim qismidir.",
    },
    {
        "name": "Oqsaroy saroyi",
        "answer": "Oqsaroy saroyi Shahrisabzda Amir Temur tomonidan qurilgan ulkan inshootdir. Uning devoriy bezaklari Temuriylar davri san'atini aks ettiradi.",
    },
    {
        "name": "Chor minor",
        "answer": "Chor minor Buxorodagi to'rt minarali madrasa bo'lib, o'zining g'aroyib me'moriy tuzilishi bilan ajralib turadi. Inshoot XIX asr boshida qurilgan.",
    },
    {
        "name": "Pahlavon Mahmud maqbarasi",
        "answer": "Pahlavon Mahmud maqbarasi Xiva shahrida joylashgan muqaddas ziyoratgohdir. U shoir va kurashchi Pahlavon Mahmud sharafiga barpo etilgan.",
    },
    {
        "name": "Savitskiy muzeyi",
        "answer": "Savitskiy muzeyi Nukusda joylashgan va avangard san'at to'plamlari bilan mashhur. Muzeyga dunyoning turli mamlakatlaridan tadqiqotchilar tashrif buyuradi.",
    },
    {
        "name": "O'zbekiston davlat san'at muzeyi",
        "answer": "O'zbekiston davlat san'at muzeyi poytaxt Toshkentda joylashgan va milliy hamda jahon san'ati namunalarini namoyish etadi. Muzey kolleksiyasi XIX-XX asr asarlarini qamrab oladi.",
    },
    {
        "name": "O'zbekiston Milliy kutubxonasi",
        "answer": "O'zbekiston Milliy kutubxonasi eng katta ilmiy va ma'rifiy axborot markazidir. Bu yerda millionlab kitob va qo'lyozmalar saqlanadi.",
    },
    {
        "name": "Oliy Majlis",
        "answer": "Oliy Majlis O'zbekiston Respublikasining oliy qonun chiqaruvchi organidir. Parlament ikki palatadan iborat bo'lib, qonunlarni qabul qiladi va nazorat qiladi.",
    },
    {
        "name": "O'zbekiston Respublikasi Konstitutsiyasi",
        "answer": "O'zbekiston Respublikasi Konstitutsiyasi davlatning asosiy qonunidir. U fuqarolarning huquq va erkinliklarini kafolatlaydi va davlat tuzilishini belgilaydi.",
    },
    {
        "name": "Davlat madhiyasi",
        "answer": "O'zbekiston davlat madhiyasi mamlakatning rasmiy ramzlaridan biri bo'lib, xalq shoiri Abdulla Oripov matni asosida yaratilgan. Madhiyaning musiqasi Mutal Burxonov tomonidan bastalangan.",
    },
    {
        "name": "O'zbek tili",
        "answer": "O'zbek tili turkiy tillar oilasiga mansub bo'lib, davlat tili maqomiga ega. Til turli shevalar va boy leksik zaxiraga ega.",
    },
    {
        "name": "Lotin yozuvi",
        "answer": "Lotin yozuvi O'zbekistonda davlat tili uchun qabul qilingan grafika tizimidir. U 1990-yillardan boshlab bosqichma-bosqich amaliyotga joriy qilinmoqda.",
    },
    {
        "name": "Lazgi raqsi",
        "answer": "Lazgi raqsi Xorazm hududiga xos jo'shqin milliy raqsdir. U tez ritmlar va energetik harakatlari bilan ajralib turadi.",
    },
    {
        "name": "Andijon polkasi",
        "answer": "Andijon polkasi xalq raqsi o'zining sho'x qadam tashlashlari bilan mashhur. Raqsda xalqning shodligi va hamjihatligi ifodalanadi.",
    },
    {
        "name": "Doira cholg'usi",
        "answer": "Doira o'zbek xalq cholg'usi bo'lib, ritmik ohanglar yaratishda qo'llaniladi. U ko'plab xalq ansambllarining ajralmas qismi hisoblanadi.",
    },
    {
        "name": "Dutor",
        "answer": "Dutor ikki torli cholg'u asbobi bo'lib, lirika kuylar ijrosida keng qo'llaniladi. Uning sadosi nafosat va muloyimlikni ifoda etadi.",
    },
    {
        "name": "Tanbur",
        "answer": "Tanbur qadimiy torli cholg'u bo'lib, maqom ijrosida muhim o'rin tutadi. U murakkab kuylarni ijro etishda mohirlik talab qiladi.",
    },
    {
        "name": "Qanun",
        "answer": "Qanun uchburchak shakldagi ko'p torli cholg'u asbobi bo'lib, Sharq musiqasida keng tarqalgan. U orkestr va ansambllarda asosiy melodiya olib boradi.",
    },
    {
        "name": "Nay",
        "answer": "Nay qadimiy puflama cholg'u bo'lib, mayin va sirli tovushlari bilan tinglovchini rom etadi. U sufiy musiqa an'analarida ham keng qo'llaniladi.",
    },
    {
        "name": "Karnay-surnay",
        "answer": "Karnay va surnay marosim va bayramlarda ishlatiladigan puflama cholg'ulardir. Ularning jarangdor sadosi tantanali kayfiyat yaratadi.",
    },
    {
        "name": "Shashmaqom",
        "answer": "Shashmaqom o'zbek va tojik xalqlarining an'anaviy maqom yo'llaridan iborat murakkab musiqiy turkumdir. U UNESCO nomoddiy meros ro'yxatiga kiritilgan.",
    },
    {
        "name": "Maqom san'ati",
        "answer": "Maqom san'ati Sharq musiqa an'analarining cho'qqisi bo'lib, kuy va ashula uyg'unligini ifoda etadi. U yuqori darajadagi ijrochilik mahoratini talab qiladi.",
    },
    {
        "name": "O'zbek milliy teatrlari",
        "answer": "O'zbek milliy teatrlari dramatik va musiqiy spektakllar orqali xalq hayotini sahnaga olib chiqadi. Ular ma'rifat va madaniyatni targ'ib etishda muhim rol o'ynaydi.",
    },
    {
        "name": "Alpomish dostoni",
        "answer": "Alpomish dostoni o'zbek xalq og'zaki ijodining durdonasi bo'lib, qahramonlik va sadoqat mavzularini yoritadi. Doston ko'plab baxshilar tomonidan ijro etib kelinadi.",
    },
    {
        "name": "Go'ro'g'li eposi",
        "answer": "Go'ro'g'li eposi turkiy xalqlarga mansub epik rivoyatlar majmuasidir. Unda adolat va erkinlik g'oyalari madh etiladi.",
    },
    {
        "name": "Avesto",
        "answer": "Avesto zardushtiylik dinining muqaddas kitobi bo'lib, qadimiy axloqiy va diniy qarashlarni o'z ichiga oladi. U Markaziy Osiyo hududida diniy madaniyat shakllanishida katta rol o'ynagan.",
    },
    {
        "name": "Toshkent metropoliteni",
        "answer": "Toshkent metropoliteni Markaziy Osiyoda qurilgan ilk metro tizimidir. Uning bekatlari o'ziga xos san'at asarlari bilan bezatilgan.",
    },
    {
        "name": "Angren",
        "answer": "Angren Toshkent viloyatidagi sanoat shahri bo'lib, ko'mir qazib olish bilan tanilgan. Shaharda energetika va kimyo korxonalari faoliyat yuritadi.",
    },
    {
        "name": "Bunyodkor stadioni",
        "answer": "Bunyodkor stadioni Toshkentdagi zamonaviy futbol arenalaridan biridir. Stadion xalqaro musobaqalar va kontsertlarga mezbonlik qiladi.",
    },
    {
        "name": "Milliy stadion",
        "answer": "O'zbekiston milliy stadioni yirik sport tadbirlari o'tkaziladigan markaziy arena hisoblanadi. U xalqaro standartlarga javob beradigan infratuzilmaga ega.",
    },
    {
        "name": "Chorsu bozori",
        "answer": "Chorsu bozori Toshkentdagi qadimiy savdo markazi bo'lib, keng assortimentdagi mahsulotlar bilan mashhur. Bozor milliy taomlar va hunarmandchilik buyumlarini topish uchun eng qulay joylardan biridir.",
    },
    {
        "name": "Siyob bozori",
        "answer": "Siyob bozori Samarqanddagi tarixiy savdo maydonidir. Bu yerda mahalliy mevalar, ziravorlar va hunarmandlar mahsulotlari sotiladi.",
    },
    {
        "name": "Urgut bozori",
        "answer": "Urgut bozori xalqaro sayyohlar sevib tashrif buyuradigan joy bo'lib, gilam va kulolchilik buyumlari bilan tanilgan. Bozor qadimiy savdo an'analarini saqlab qolgan.",
    },
    {
        "name": "O'zbekiston Milliy universiteti",
        "answer": "O'zbekiston Milliy universiteti mamlakatdagi eng qadimiy oliy ta'lim muassasalaridan biridir. U ko'plab fan sohalarida malakali mutaxassislar tayyorlaydi.",
    },
    {
        "name": "Toshkent davlat texnika universiteti",
        "answer": "Toshkent davlat texnika universiteti muhandislik va texnologiya yo'nalishlariga ixtisoslashgan. Universitet sanoat uchun innovatsion kadrlar yetkazib beradi.",
    },
    {
        "name": "Samarqand davlat universiteti",
        "answer": "Samarqand davlat universiteti 1927 yilda tashkil topgan va ko'plab ilmiy maktablarga ega. U mintaqadagi ilmiy izlanishlar markazlaridan biri hisoblanadi.",
    },
    {
        "name": "Andijon davlat universiteti",
        "answer": "Andijon davlat universiteti Farg'ona vodiysida joylashgan yirik oliygohdir. Universitet pedagogika, filologiya va tabiiy fanlar bo'yicha mutaxassislar tayyorlaydi.",
    },
    {
        "name": "Toshkent axborot texnologiyalari universiteti",
        "answer": "Toshkent axborot texnologiyalari universiteti telekommunikatsiya va IT sohasida yetakchi oliy ta'lim muassasasi hisoblanadi. U zamonaviy laboratoriyalar va startap ekotizimini rivojlantiradi.",
    },
    {
        "name": "Inha universiteti",
        "answer": "Toshkentdagi Inha universiteti Koreya bilan hamkorlikda tashkil etilgan zamonaviy texnologik oliygohdir. U axborot texnologiyalari va logistika bo'yicha xalqaro dasturlarni taklif etadi.",
    },
    {
        "name": "Prezident maktablari",
        "answer": "Prezident maktablari iqtidorli o'quvchilar uchun tashkil etilgan zamonaviy ta'lim maskanlaridir. Ular STEAM yo'nalishidagi chuqurlashtirilgan dasturlarni amalga oshiradi.",
    },
    {
        "name": "IT Park Uzbekistan",
        "answer": "IT Park Uzbekistan mamlakatning raqamli iqtisodiyotini rivojlantirishga qaratilgan innovatsion markazdir. Park startaplar va IT kompaniyalarini qo'llab-quvvatlash uchun platforma yaratadi.",
    },
    {
        "name": "Yangi O'zbekiston bog'i",
        "answer": "Yangi O'zbekiston bog'i Toshkentdagi katta rekreatsion hudud bo'lib, zamonaviy dam olish infratuzilmasiga ega. Bog'da oilaviy hordiq va madaniy tadbirlar uchun sharoit yaratilgan.",
    },
    {
        "name": "Mustaqillik maydoni",
        "answer": "Mustaqillik maydoni Toshkentning markazida joylashgan ramziy maskandir. Bu yerda davlatning asosiy bayram tantanalari o'tkaziladi.",
    },
    {
        "name": "Amir Temur xiyoboni",
        "answer": "Amir Temur xiyoboni poytaxtning tarixiy va madaniy markazlaridan biridir. Xiyobonda Amir Temur haykali va ko'plab muzeylar joylashgan.",
    },
    {
        "name": "O'zbekiston davlat bayrog'i",
        "answer": "O'zbekiston davlat bayrog'i ko'k, oq va yashil ranglardan iborat bo'lib, mustaqillik ramzidir. Bayroqdagi oy va yulduzlar xalqning tarixiy va ma'naviy qadriyatlarini ifodalaydi.",
    },
]


def build_dataset() -> list[dict[str, str]]:
    """Create the list of question and answer entries."""
    records: list[dict[str, str]] = []
    idx = 1
    for fact in FACTS:
        for template in QUESTION_TEMPLATES:
            question = template.format(name=fact["name"])
            records.append(
                {
                    "id": idx,
                    "topic": fact["name"],
                    "input_text": fact["answer"],
                    "target_text": question,
                }
            )
            idx += 1
    return records


def main() -> None:
    dataset = build_dataset()
    if len(dataset) != len(FACTS) * len(QUESTION_TEMPLATES):
        raise ValueError("Dataset size mismatch")

    output_path = BASE_DIR / "data" / "uzbek_qa_1000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
