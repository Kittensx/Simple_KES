from modules.sd_simple_kes.schedulers.karras_advanced_scheduler import get_sigmas_karras
from modules.sd_simple_kes.schedulers.exponential_advanced_scheduler import get_sigmas_exponential
from modules.sd_simple_kes.schedulers.geometric_advanced_scheduler import get_sigmas_geometric
from modules.sd_simple_kes.schedulers.harmonic_advanced_scheduler import get_sigmas_harmonic
from modules.sd_simple_kes.schedulers.logarithmic_advanced_scheduler import get_sigmas_logarithmic
from modules.sd_simple_kes.schedulers.euler_advanced_scheduler import get_sigmas_euler, get_sigmas_euler_advanced


scheduler_registry = {
    'karras': get_sigmas_karras,
    'exponential': get_sigmas_exponential,
    'geometric': get_sigmas_geometric,
    'harmonic': get_sigmas_harmonic,
    'logarithmic': get_sigmas_logarithmic,
    'euler': get_sigmas_euler,
    'euler_advanced': get_sigmas_euler_advanced
    # Add more here - ensure methods are added to get_sigmas then update imports #also update simple_kes
}