#include "integrals.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <string>
#include "../config.h"
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"
#include "dooh_util.h"

void Integrals::load() {
  integrals_1b.set_storage(Config::get<bool>("hash_integrals", true));
  integrals_2b.set_storage(Config::get<bool>("hash_integrals", true));
  const std::string& cache_filename = "integrals_cache.dat";
  if (Config::get<bool>("load_integrals_cache", false) && load_from_cache(cache_filename)) return;
  read_fcidump();
  Timer::checkpoint("load fcidump");
  if (!Config::get<bool>("hash_integrals", true)) {
    printf("Vector storage in use.\n");
    size_t num_filled = integrals_2b.num_elements() + integrals_1b.num_elements();
    double percent = 100 * 8 * (float)num_filled / (n_orbs * n_orbs * n_orbs * n_orbs);
    printf("%.2f%% of vector elements are filled.\n", percent);
  }
  generate_det_hf();
  const auto& orb_energies = get_orb_energies();
  reorder_orbs(orb_energies);
  Timer::checkpoint("reorder orbitals");
  save_to_cache(cache_filename);
}

void Integrals::read_fcidump() {
  std::ifstream fcidump("FCIDUMP");
  if (!fcidump.good()) {
    throw new std::runtime_error("cannot open FCIDUMP");
  }

  // Read head.
  std::regex words("([\\-]?[^\\s,=\\-]+)");
  std::string line;
  enum class State { NONE, ORBSYM, END };
  State state = State::NONE;
  std::vector<int> orb_syms_raw;
  while (!fcidump.eof()) {
    std::getline(fcidump, line);
    const auto& words_begin = std::sregex_iterator(line.begin(), line.end(), words);
    const auto& words_end = std::sregex_iterator();
    for (auto it = words_begin; it != words_end; it++) {
      const std::string match = it->str();
      if (match == "NORB" || match == "NORBS") {
        it++;
        n_orbs = std::stoul(it->str());
        if (Parallel::is_master()) printf("n_orbs: %u\n", n_orbs);
        orb_syms_raw.reserve(n_orbs);
      } else if (match == "NELEC") {
        it++;
        n_elecs = std::stoul(it->str());
        if (Parallel::is_master()) printf("n_elecs (from FCIDUMP): %u\n", n_elecs);
      } else if (match == "ORBSYM") {
        state = State::ORBSYM;
        if (Parallel::is_master()) printf("orb_sym: ");
      } else if (state == State::ORBSYM) {
        const int orb_sym_raw = std::stoi(match);
        orb_syms_raw.push_back(orb_sym_raw);
        if (Parallel::is_master()) printf("%d ", orb_sym_raw);
        if (orb_syms_raw.size() == n_orbs) {
          state = State::NONE;
          if (Parallel::is_master()) printf("\n");
        }
      } else if (match == "&END" || match == "/") {
        state = State::END;
      }
    }
    if (state == State::END) break;
  }
  orb_sym = get_adams_syms(orb_syms_raw);

  // Read integrals.
  double integral;
  unsigned p, q, r, s;
  while (true) {
    fcidump >> integral >> p >> q >> r >> s;
    if (fcidump.eof()) break;
    if (std::abs(integral) < 1.0e-9) continue;
    raw_integrals.push_back(Hpqrs(integral, p, q, r, s));
  }
  fcidump.close();

  energy_core = 0.0;
  for (const auto& item : raw_integrals) {
    const unsigned p = item.p;
    const unsigned q = item.q;
    const unsigned r = item.r;
    const unsigned s = item.s;
    const double integral = item.H;
    if (p == q && q == r && r == s && s == 0) {
      energy_core = integral;
    } else if (r == s && s == 0) {
      integrals_1b.set(combine2(p - 1, q - 1), integral, [&](double& a, const double& b) {
        if (std::abs(a) < std::abs(b)) a = b;
      });
    } else {
      integrals_2b.set(
          combine4(p - 1, q - 1, r - 1, s - 1), integral, [&](double& a, const double& b) {
            if (std::abs(a) < std::abs(b)) a = b;
          });
    }
  }
}

std::vector<unsigned> Integrals::get_adams_syms(const std::vector<int>& orb_syms_raw) const {
  std::vector<unsigned> adams_syms;
  bool has_negative = false;
  for (const auto& orb : orb_syms_raw) {
    if (orb < 0) {
      has_negative = true;
      break;
    }
  }
  if (!has_negative) {
    adams_syms.assign(orb_syms_raw.begin(), orb_syms_raw.end());
    return adams_syms;
  }

  // Convert to Adam's notation from Sandeep's notation.
  adams_syms.resize(orb_syms_raw.size());
  for (size_t i = 0; i < orb_syms_raw.size(); i++) {
    const int orb_sym = orb_syms_raw[i];
    if (orb_sym == 1 || orb_sym == 2) {
      adams_syms[i] = orb_sym;
      continue;
    }
    const unsigned a = std::abs(orb_sym) >> 1;
    const unsigned b = (std::abs(orb_sym) + 1) >> 1;
    unsigned orb_sym_new = a + 3 * b - 8;
    if (orb_sym < 0) orb_sym_new += 2;
    adams_syms[i] = orb_sym_new;
  }
  if (Parallel::is_master()) {
    printf("Convert to Adam's notation:\n");
    for (size_t i = 0; i < adams_syms.size(); i++) {
      printf("%u ", adams_syms[i]);
    }
    printf("\n");
  }

  return adams_syms;
}

void Integrals::generate_det_hf() {
  det_hf = Det();
  n_up = Config::get<unsigned>("n_up");
  n_dn = Config::get<unsigned>("n_dn");
  if (Parallel::is_master() && n_up + n_dn != n_elecs) {
    printf("WARNING: n_up and n_dn from config doesn't match FCIDUMP\n");
  }
  n_elecs = n_up + n_dn;
  std::vector<unsigned> irreps =
      Config::get<std::vector<unsigned>>("chem/irreps", std::vector<unsigned>());
  std::vector<unsigned> occs_up =
      Config::get<std::vector<unsigned>>("occs_up", std::vector<unsigned>());
  std::vector<unsigned> occs_dn =
      Config::get<std::vector<unsigned>>("occs_dn", std::vector<unsigned>());
  explicit_orbs = false;
  if (occs_up.size() > 0 || occs_dn.size() > 0) {
    if (occs_up.size() != n_up) throw std::invalid_argument("occs_up does not match n_up");
    if (occs_dn.size() != n_dn) throw std::invalid_argument("occs_dn does not match n_dn");
    for (unsigned i = 0; i < n_up; i++) det_hf.up.set(occs_up[i]);
    for (unsigned i = 0; i < n_dn; i++) det_hf.dn.set(occs_dn[i]);
    explicit_orbs = true;
  } else if (irreps.size() == 0) {
    // Fill lowest.
    for (unsigned i = 0; i < n_up; i++) det_hf.up.set(i);
    bool allow_doubly_occupy = Config::get<bool>("allow_doubly_occupy_hf", true);
    if (allow_doubly_occupy) {
      for (unsigned i = 0; i < n_dn; i++) det_hf.dn.set(i);
    } else {
      assert(n_up + n_up <= n_orbs);
      for (unsigned i = 0; i < n_dn; i++) det_hf.dn.set(i + n_up);
    }
  } else {
    // Fill according to irreps.
    std::vector<unsigned> irrep_occs_up = Config::get<std::vector<unsigned>>("chem/irrep_occs_up");
    std::vector<unsigned> irrep_occs_dn = Config::get<std::vector<unsigned>>("chem/irrep_occs_dn");
    assert(std::accumulate(irrep_occs_up.begin(), irrep_occs_up.end(), 0u) == n_up);
    assert(std::accumulate(irrep_occs_dn.begin(), irrep_occs_dn.end(), 0u) == n_dn);
    const unsigned n_irreps = irreps.size();
    for (unsigned i = 0; i < n_irreps; i++) {
      const unsigned irrep = irreps[i];
      unsigned irrep_occ_up = irrep_occs_up[i];
      unsigned irrep_occ_dn = irrep_occs_dn[i];
      for (unsigned j = 0; j < n_orbs; j++) {
        if (orb_sym[j] != irrep) continue;
        if (irrep_occ_up > 0) {
          det_hf.up.set(j);
          irrep_occ_up--;
        } else {
          det_hf.up.unset(j);
        }
        if (irrep_occ_dn > 0) {
          det_hf.dn.set(j);
          irrep_occ_dn--;
        } else {
          det_hf.dn.unset(j);
        }
        if (irrep_occ_up == 0 && irrep_occ_dn == 0 && j >= n_up && j >= n_dn) break;
      }
      if (irrep_occ_up > 0 || irrep_occ_dn > 0) {
        throw std::runtime_error("unable to construct hf with given irrep");
      }
    }
    explicit_orbs = true;
  }
  if (Parallel::is_master()) {
    printf("HF det up: ");
    for (const unsigned orb : det_hf.up.get_occupied_orbs()) printf("%u ", orb);
    printf("\nHF det dn: ");
    for (const unsigned orb : det_hf.dn.get_occupied_orbs()) printf("%u ", orb);
    printf("\n");
  }
}

std::vector<double> Integrals::get_orb_energies() const {
  std::vector<double> orb_energies(n_orbs);
  const auto& orbs_up = det_hf.up.get_occupied_orbs();
  const auto& orbs_dn = det_hf.dn.get_occupied_orbs();
  for (unsigned i = 0; i < n_orbs; i++) {
    orb_energies[i] = get_1b(i, i);
    double energy_direct = 0;
    double energy_exchange = 0;
    for (const unsigned orb : orbs_up) {
      if (orb == i) {
        energy_direct += get_2b(i, i, orb, orb);
      } else {
        energy_direct += 2 * get_2b(i, i, orb, orb);
        energy_exchange -= get_2b(i, orb, orb, i);
      }
    }
    for (const unsigned orb : orbs_dn) {
      if (orb == i) {
        energy_direct += get_2b(i, i, orb, orb);
      } else {
        energy_direct += 2 * get_2b(i, i, orb, orb);
        energy_exchange -= get_2b(i, orb, orb, i);
      }
    }
    orb_energies[i] += 0.5 * (energy_direct + energy_exchange);
  }
  return orb_energies;
}

void Integrals::reorder_orbs(const std::vector<double>& orb_energies) {
  orb_order.resize(n_orbs);
  orb_order_inv.resize(n_orbs);
  std::iota(orb_order.begin(), orb_order.end(), 0);
  if (Config::get<bool>("reorder_orbs", !explicit_orbs)) {
    std::stable_sort(orb_order.begin(), orb_order.end(), [&](const unsigned a, const unsigned b) {
      return orb_energies[a] < orb_energies[b] - Util::EPS;
    });
  } else {
    if (Parallel::is_master()) {
      printf("Reorder skipped.\n");
    }
  }

  // Reorder orb_sym.
  std::vector<unsigned> orb_syms_new(n_orbs);
  for (unsigned i = 0; i < n_orbs; i++) {
    orb_syms_new[i] = orb_sym[orb_order[i]];
  }
  orb_sym = std::move(orb_syms_new);

  if (Parallel::is_master()) printf("Orbitals energy:\n");
  for (unsigned i = 0; i < n_orbs; i++) {
    orb_order_inv[orb_order[i]] = i;
    const unsigned ori_id = orb_order[i];
    const double orb_energy = orb_energies[ori_id];
    if (Parallel::is_master()) {
      printf("#%3u: E = %16.12f, E_1b = %16.12f, sym = %2u, origin #%3u\n", i, orb_energy, get_1b(ori_id, ori_id), orb_sym[i], ori_id);
    }
  }

  // Update HF det.
  Det det_hf_new = Det();
  for (unsigned i = 0; i < n_orbs; i++) {
    if (det_hf.up.has(orb_order[i])) {
      det_hf_new.up.set(i);
    } else {
      det_hf_new.up.unset(i);
    }
    if (det_hf.dn.has(orb_order[i])) {
      det_hf_new.dn.set(i);
    } else {
      det_hf_new.dn.unset(i);
    }
  }
  det_hf = std::move(det_hf_new);
  if (Parallel::is_master()) {
    printf("HF det up: ");
    for (const unsigned orb : det_hf.up.get_occupied_orbs()) printf("%u ", orb);
    printf("\nHF det dn: ");
    for (const unsigned orb : det_hf.dn.get_occupied_orbs()) printf("%u ", orb);
    printf("\n");
  }

  integrals_1b.clear();
  integrals_2b.clear();
  for (const auto& item : raw_integrals) {
    const unsigned p = item.p;
    const unsigned q = item.q;
    const unsigned r = item.r;
    const unsigned s = item.s;
    const double integral = item.H;
    if (p == q && q == r && r == s && s == 0) {
      continue;
    } else if (r == s && s == 0) {
      integrals_1b.set(
          combine2(orb_order_inv[p - 1], orb_order_inv[q - 1]),
          integral,
          [&](double& a, const double& b) {
            if (std::abs(a) < std::abs(b)) a = b;
          });
    } else {
      integrals_2b.set(
          combine4(
              orb_order_inv[p - 1],
              orb_order_inv[q - 1],
              orb_order_inv[r - 1],
              orb_order_inv[s - 1]),
          integral,
          [&](double& a, const double& b) {
            if (std::abs(a) < std::abs(b)) a = b;
          });
    }
  }
  raw_integrals.clear();
  raw_integrals.shrink_to_fit();
}

void Integrals::set_point_group(const PointGroup& group_name) {
  point_group = group_name;
}

double Integrals::get_1b(const unsigned p, const unsigned q) const {
  const size_t combined = combine2(p, q);
  return integrals_1b.get(combined, 0.0);
}

double Integrals::get_2b(
    const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  unsigned p_sym = orb_sym[p];
  unsigned q_sym = orb_sym[q];
  unsigned r_sym = orb_sym[r];
  unsigned s_sym = orb_sym[s];
  int gu;
  if ((point_group == PointGroup::Dooh) || (point_group == PointGroup::Coov)) {
    if ((DoohUtil::get_lz(p_sym, gu) + DoohUtil::get_lz(r_sym, gu)) != (DoohUtil::get_lz(q_sym, gu) + DoohUtil::get_lz(s_sym, gu))) 
    return 0.;
  }
  const size_t combined = combine4(p, q, r, s);
  return integrals_2b.get(combined, 0.0);
}

size_t Integrals::combine2(const size_t a, const size_t b) {
  if (a > b) {
    return (a * (a + 1)) / 2 + b;
  } else {
    return (b * (b + 1)) / 2 + a;
  }
}

size_t Integrals::combine4(const size_t a, const size_t b, const size_t c, const size_t d) {
  const size_t ab = combine2(a, b);
  const size_t cd = combine2(c, d);
  return combine2(ab, cd);
}

bool Integrals::load_from_cache(const std::string& filename) {
  std::ifstream file(filename, std::ifstream::binary);
  if (!file) return false;
  hps::from_stream<Integrals>(file, *this);
  if (Parallel::get_proc_id() == 0) {
    printf("Loaded FCIDUMP cache from: %s\n", filename.c_str());
  }
  return true;
}

void Integrals::save_to_cache(const std::string& filename) const {
  if (Parallel::get_proc_id() == 0) {
    std::ofstream file(filename, std::ofstream::binary);
    hps::to_stream(*this, file);
    printf("FCIDUMP cache saved to: %s\n", filename.c_str());
  }
}

void Integrals::dump_integrals(const char* filename) const {
  Timer::start("create new FCIDUMP");
  bool is_infinity_group = (point_group == PointGroup::Dooh) || (point_group == PointGroup::Coov);

  if (Parallel::is_master()) {
    FILE* pFile;
    pFile = fopen(filename, "w");

    // Header
    fprintf(pFile, " &FCI NORB=%d, NELEC=%d, MS2=%d,\n", n_orbs, n_elecs, 0);
    fprintf(pFile, "ORBSYM=");
    for (unsigned i = 0; i < n_orbs; i++) {
      fprintf(pFile, "  %d", orb_sym[orb_order_inv[i]]);
    }
    if (is_infinity_group) fprintf(pFile, "\ninfinity group");
    fprintf(pFile, "\nISYM=1\n&END\n");

    double integral_value;
    unsigned p, q, r, s;

    // Two-body integrals
    if (is_infinity_group) { // 4-fold symmetry
      for (p = 0; p < n_orbs; p++) {
        for (q = 0; q <= p; q++) {
          for (r = 0; r <= p; r++) {
            for (s = 0; s < n_orbs; s++) {
  	      if ((p == r) && (q < s)) continue;
              integral_value = get_2b(p, q, r, s);
              if (std::abs(integral_value) > 1e-9) {
                fprintf(
                    pFile,
                    " %19.12E %3d %3d %3d %3d\n",
                    integral_value,
                    orb_order[p] + 1,
                    orb_order[q] + 1,
                    orb_order[r] + 1,
                    orb_order[s] + 1);
              }
            }  // s
          }  // r
        }  // q
      }  // p
    } else { // 8-fold symmetry
      for (p = 0; p < n_orbs; p++) {
        for (q = 0; q <= p; q++) {
          for (r = 0; r <= p; r++) {
            for (s = 0; s <= r; s++) {
              if ((p == r) && (q < s)) continue;
              integral_value = get_2b(p, q, r, s);
              if (std::abs(integral_value) > 1e-9) {
                fprintf(
                    pFile,
                    " %19.12E %3d %3d %3d %3d\n",
                    integral_value,
                    orb_order[p] + 1,
                    orb_order[q] + 1,
                    orb_order[r] + 1,
                    orb_order[s] + 1);
              }
            }  // s
          }  // r
        }  // q
      }  // p
    }

    // One-body integrals
    for (p = 0; p < n_orbs; p++) {
      for (q = 0; q <= p; q++) {
        integral_value = get_1b(p, q);
        if (std::abs(integral_value) > 1e-9) {
          fprintf(
              pFile,
              " %19.12E %3d %3d %3d %3d\n",
              integral_value,
              orb_order[p] + 1,
              orb_order[q] + 1,
              0,
              0);
        }
      }
    }

    // Nuclear-nuclear energy
    fprintf(pFile, " %19.12E %3d %3d %3d %3d\n", energy_core, 0, 0, 0, 0);

    fclose(pFile);
  }

  Timer::end();
}
