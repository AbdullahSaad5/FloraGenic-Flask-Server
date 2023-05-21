from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import jsonify
from PIL import Image




classes = [
    'Gomphocarpus_physocarpus',
    'Moehringia_ciliata',
    'Phedimus_aizoon',
    'Sedum_nussbaumerianum',
    'Trifolium_badium',
    'Telekia_speciosa',
    'Oncostema_peruviana',
    'Angelica_archangelica',
    'Sedum_kamtschaticum',
    'Duchesnea_indica',
    'Erechtites_hieraciifolius',
    'Holodiscus_discolor',
    'Trifolium_alpestre',
    'Hypericum_patulum',
    'Tradescantia_pallida',
    'Peperomia_pellucida',
    'Tradescantia_zebrina',
    'Sedum_hispanicum',
    'Trifolium_fragiferum',
    'Cytinus_hypocistis',
    'Mertensia_virginica',
    'Hypericum_tetrapterum',
    'Melilotus_officinalis',
    'Nephrolepis_biserrata',
    'Clethra_alnifolia',
    'Acacia_retinodes',
    'Lactuca_plumieri',
    'Phedimus_spurius',
    'Cirsium_palustre',
    'Lactuca_viminea',
    'Trifolium_hybridum',
    'Calendula_arvensis',
    'Alocasia_cucullata',
    'Trifolium_resupinatum',
    'Loropetalum_chinense',
    'Pilosocereus_pachycladus',
    'Selenicereus_anthonyanus',
    'Daphne_cneorum',
    'Ophrys_speculum',
    'Anemone_apennina',
    'Narthecium_ossifragum',
    'Acalypha_wilkesiana',
    'Anemone_narcissiflora',
    'Sedum_spathulifolium',
    'Cereus_jamacaru',
    'Sedum_clavatum',
    'Adonis_vernalis',
    'Smilax_rotundifolia',
    'Barbarea_verna',
    'Freesia_refracta',
    'Anthericum_ramosum',
    'Freesia_alba',
    'Peperomia_magnoliifolia',
    'Lupinus_arboreus',
    'Ophrys_bertolonii',
    'Sedum_pachyphyllum',
    'Morinda_citrifolia',
    'Atocion_rupestre',
    'Centranthus_calcitrapae',
    'Erucastrum_incanum',
    'Aphelandra_squarrosa',
    'Fragaria_x_ananassa',
    'Trifolium_rubens',
    'Neotinea_tridentata',
    'Viscaria_vulgaris',
    'Sedum_forsterianum',
    'Conoclinium_coelestinum',
    'Gynura_aurantiaca',
    'Falcaria_vulgaris',
    'Tradescantia_sillamontana',
    'Couroupita_guianensis',
    'Lamium_maculatum',
    'Trifolium_aureum',
    'Acacia_longifolia',
    'Duchesnea_indica',
    'Honckenya_peploides',
    'Pelargonium_x_hortorum',
    'Sedum_anglicum',
    'Cirsium_rivulare',
    'Lagenaria_siceraria',
    'Sedum_sarmentosum',
    'Ophrys_fusca',
    'Ophrys_lutea',
    'Ophrys_insectifera',
    'Myosoton_aquaticum',
    'Lithops_pseudotruncatella',
    'Hypericum_hirsutum',
    'Centranthus_angustifolius',
    'Acacia_mearnsii',
    'Hypericum_hircinum',
    'Adenostyles_alliariae',
    'Fragaria_viridis',
    'Acalypha_virginica',
    'Cirsium_spinosissimum',
    'Peperomia_argyreia',
    'Adonis_annua',
    'Carthamus_lanatus',
    'Anemone_canadensis',
    'Hyoscyamus_albus',
    'Tradescantia_ohiensis',
    'Hyoscyamus_niger',
    'Aralia_nudicaulis',
    'Secale_cereale',
    'Ophrys_tenthredinifera',
    'Casuarina_cunninghamiana',
    'Asystasia_gangetica',
    'Cirsium_erisithales',
    'Ophrys_passionis',
    'Mussaenda_erythrophylla',
    'Papaver_hybridum',
    'Carthamus_tinctorius',
    'Hyoseris_radiata',
    'Cirsium_acaulon',
    'Maianthemum_canadense',
    'Empetrum_nigrum',
    'Smilax_bona-nox',
    'Sedum_sexangulare',
    'Peperomia_obtusifolia',
    'Acalypha_hispida',
    'Meum_athamanticum',
    'Hypericum_humifusum',
    'Cereus_hexagonus',
    'Pelargonium_x_hybridum',
    'Trifolium_montanum',
    'Althaea_cannabina',
    'Ophrys_aranifera',
    'Peperomia_caperata',
    'Epipactis_palustris',
    'Sedum_burrito',
    'Breynia_disticha',
    'Diascia_rigescens',
    'Mussaenda_philippica',
    'Cirsium_heterophyllum',
    'Papaver_argemone',
    'Trifolium_medium',
    'Maianthemum_racemosum',
    'Pelargonium_peltatum',
    'Anemone_vernalis',
    'Peperomia_serpens',
    'Liriope_muscari',
    'Chaerophyllum_hirsutum'
 ]

def plant_species(image_path, model):
    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    # Make prediction using model loaded from disk as per the data. 
    predictions = model.predict(x)
    # Take the first value of prediction
    probability = np.max(predictions[0])
    index = np.argmax(predictions[0])
    if(probability < 0.5):
        return None
    else:
        return ({'Species': classes[index], 'Probability': str(probability)})