#include "integrals.h"

#include "../config.h"
#include "../parallel.h"

void Integrals::load() {
  read_fcidump();
  generate_det_hf();
  const auto& orb_energies = get_orb_energies();
  // reorder_orbs(orb_energies);
}

void Integrals::read_fcidump() {
  FILE* fcidump = fopen("FCIDUMP", "r");
  if (!fcidump) throw new std::runtime_error("FCIDUMP not found");
  fscanf(fcidump, " %*4s %*5s %u %*7s %u %*s %*7s", &n_orbs, &n_elecs);
  printf("n_orbs: %u\nn_elecs: %u\n", n_orbs, n_elecs);
  std::vector<int> orb_syms_raw(n_orbs);
  printf("orb_syms_raw: ");
  for (unsigned i = 0; i < n_orbs; i++) {
    fscanf(fcidump, "%u%*c", &orb_syms_raw[i]);
    printf("%d ", orb_syms_raw[i]);
  }
  printf("\n");
  orb_syms = get_adams_syms(orb_syms_raw);
  fscanf(fcidump, "%*s %*s");
  double integral;
  unsigned p, q, r, s;
  while (fscanf(fcidump, "%lf %u %u %u %u", &integral, &p, &q, &r, &s) != EOF) {
    if (p == q && q == r && r == s && s == 0) {
      energy_core = integral;
    } else if (r == s && s == 0) {
      integrals_1b[combine2(p - 1, q - 1)] = integral;
    } else {
      integrals_2b[combine4(p - 1, q - 1, r - 1, s - 1)] = integral;
    }
    raw_integrals.push_back(std::make_tuple(p, q, r, s, integral));
  }
  fclose(fcidump);
}

std::vector<unsigned> Integrals::get_adams_syms(const std::vector<int>& orb_syms_raw) const {
  std::vector<unsigned> adams_syms;
  const auto& point_group = Config::get<std::string>("chem.point_group");
  if (point_group != "dih") {
    adams_syms.assign(orb_syms_raw.begin(), orb_syms_raw.end());
    return adams_syms;
  }

  // Convert to Adam's notation from Sandeep's notation.
}

void Integrals::generate_det_hf() {
  std::vector<unsigned> irreps = Config::get<std::vector<unsigned>>("chem.irreps");
  std::vector<unsigned> irrep_occs_up = Config::get<std::vector<unsigned>>("chem.irrep_occs_up");
  std::vector<unsigned> irrep_occs_dn = Config::get<std::vector<unsigned>>("chem.irrep_occs_dn");
  unsigned n_up = Config::get<unsigned>("n_up");
  unsigned n_dn = Config::get<unsigned>("n_dn");
  assert(n_up + n_dn == n_elecs);
  det_hf = Det(n_up, n_dn);
  const unsigned n_irreps = irreps.size();
  for (unsigned i = 0; i < n_irreps; i++) {
    const unsigned irrep = irreps[i];
    unsigned irrep_occ_up = irrep_occs_up[i];
    unsigned irrep_occ_dn = irrep_occs_dn[i];
    for (unsigned j = 0; j < n_orbs; j++) {
      if (orb_syms[j] != irrep) continue;
      if (irrep_occ_up > 0) {
        det_hf.up.set(j);
        irrep_occ_up--;
      }
      if (irrep_occ_dn > 0) {
        det_hf.dn.set(j);
        irrep_occ_dn--;
      }
      if (irrep_occ_up == 0 && irrep_occ_dn == 0) break;
    }
  }
  printf("HF det up: ");
  for (const unsigned orb : det_hf.up.get_occupied_orbs()) printf("%u ", orb);
  printf("\nHF det dn: ");
  for (const unsigned orb : det_hf.dn.get_occupied_orbs()) printf("%u ", orb);
  printf("\n");
}

std::vector<double> Integrals::get_orb_energies() const {
  std::vector<double> orb_energies(n_orbs);
  const auto& orbs_up = det_hf.up.get_occupied_orbs();
  const auto& orbs_dn = det_hf.dn.get_occupied_orbs();
  for (unsigned i = 0; i < n_orbs; i++) {
    orb_energies[i] = get_integral_1b(i, i);
    double energy_direct = 0;
    double energy_exchange = 0;
    for (const unsigned orb : orbs_up) {
      if (orb == i) {
        energy_direct += get_integral_2b(i, i, orb, orb);
      } else {
        energy_direct += 2 * get_integral_2b(i, i, orb, orb);
        energy_exchange -= get_integral_2b(i, orb, orb, i);
      }
    }
    for (const unsigned orb : orbs_dn) {
      if (orb == i) {
        energy_direct += get_integral_2b(i, i, orb, orb);
      } else {
        energy_direct += 2 * get_integral_2b(i, i, orb, orb);
        energy_exchange -= get_integral_2b(i, orb, orb, i);
      }
    }
    orb_energies[i] += 0.5 * (energy_direct + energy_exchange);
    printf("orb %u: %lf\n", i, orb_energies[i]);
  }
  return orb_energies;
}

void Integrals::reorder_orbs(const std::vector<double>& orb_energies) {}

double Integrals::get_integral_1b(const unsigned p, const unsigned q) const {
  const size_t combined = combine2(p, q);
  if (integrals_1b.count(combined) == 1) return integrals_1b.at(combined);
  return 0.0;
}

double Integrals::get_integral_2b(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  const size_t combined = combine4(p, q, r, s);
  if (integrals_2b.count(combined) == 1) return integrals_2b.at(combined);
  return 0.0;
}

size_t Integrals::combine2(const size_t a, const size_t b) const {
  if (a > b) {
    return (a * (a + 1)) / 2 + b;
  } else {
    return (b * (b + 1)) / 2 + a;
  }
}

size_t Integrals::combine4(const size_t a, const size_t b, const size_t c, const size_t d) const {
  const size_t ab = combine2(a, b);
  const size_t cd = combine2(c, d);
  return combine2(ab, cd);
}