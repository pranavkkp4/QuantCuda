// TC-JSN-001..007: JSON input validation and normalization (P1).
#include "quantcuda/json_config.hpp"
#include "test_util.hpp"

using namespace quantcuda;

namespace {
const char* GOOD = R"json({
  "simulation": {"paths": 50000, "time_steps": 52, "seed": 7,
                 "precision": "mixed", "backend": "openmp"},
  "market": {
    "risk_free_rate": 0.0425,
    "assets": [
      {"symbol": "A", "spot": 100.0, "volatility": 0.20, "dividend_yield": 0.01},
      {"symbol": "B", "spot": 75.0,  "volatility": 0.27}
    ],
    "correlation": [[1.0, 0.65], [0.65, 1.0]]
  },
  "portfolio": [
    {"id": "C1", "type": "european_call", "underlying": "A",
     "strike": 105.0, "maturity": 1.0, "quantity": 100},
    {"id": "A1", "type": "asian_call", "underlying": "B",
     "strike": 80.0, "maturity": 0.5, "quantity": 50},
    {"id": "B1", "type": "barrier_up_out_put", "underlying": "A",
     "strike": 90.0, "maturity": 0.75, "quantity": 20, "barrier": 150.0}
  ],
  "risk": {"scenarios": 20000, "confidence_levels": [0.95, 0.99]}
})json";
}

QC_TEST(json_config, parses_valid_document) {
    EngineInput in = load_engine_input(GOOD);
    QC_CHECK(in.simulation.paths == 50'000);
    QC_CHECK(in.simulation.time_steps == 52);
    QC_CHECK(in.simulation.seed == 7);
    QC_CHECK(in.simulation.precision == Precision::Mixed);
    QC_CHECK(in.simulation.backend == Backend::OpenMP);
    QC_CHECK(in.market.assets.size() == 2);
    QC_CHECK(in.portfolio.options.size() == 3);
    QC_CHECK(in.portfolio.options[2].type == OptionType::BarrierUpOutPut);
    QC_CHECK_NEAR(in.portfolio.options[2].barrier, 150.0, 1e-12);
    QC_CHECK(in.risk.confidence_levels.size() == 2);

    std::string max_seed_doc = GOOD;
    const std::string old_seed = "\"seed\": 7";
    max_seed_doc.replace(max_seed_doc.find(old_seed), old_seed.size(),
                         "\"seed\": 18446744073709551615");
    EngineInput max_seed = load_engine_input(max_seed_doc);
    QC_CHECK(max_seed.simulation.seed == 18446744073709551615ull);
}

QC_TEST(json_config, rejects_garbage) {
    QC_CHECK_THROWS(load_engine_input("{not json"));
    QC_CHECK_THROWS(load_engine_input(""));
    QC_CHECK_THROWS(load_engine_input("[]"));
    QC_CHECK_THROWS(load_engine_input(std::string(8u * 1024u * 1024u + 1u, ' ')));

    // Nlohmann parse/range failures are normalized to the documented public
    // std::runtime_error contract.
    bool got_runtime_error = false;
    try {
        (void)load_engine_input(R"({"market":{"risk_free_rate":1e999,"assets":[]},"portfolio":[]})");
    } catch (const std::runtime_error&) {
        got_runtime_error = true;
    } catch (...) {
    }
    QC_CHECK(got_runtime_error);
}

QC_TEST(json_config, rejects_missing_sections) {
    QC_CHECK_THROWS(load_engine_input(R"({"simulation": {}})"));
    QC_CHECK_THROWS(load_engine_input(R"({"market": {"risk_free_rate": 0.05, "assets": []}})"));
    QC_CHECK_THROWS(load_engine_input(R"({"simulation": [], "market": {}, "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({"risk": [], "market": {}, "portfolio": []})"));
}

QC_TEST(json_config, rejects_invalid_numbers) {
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": -1.0, "volatility": 0.2}]},
      "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 100.0, "volatility": -0.2}]},
      "portfolio": []})"));
}

QC_TEST(json_config, rejects_unknown_underlying_and_type) {
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 100.0, "volatility": 0.2}]},
      "portfolio": [{"id": "bad_underlying", "type": "european_call", "underlying": "ZZZ",
                     "strike": 100.0, "maturity": 1.0}]})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 100.0, "volatility": 0.2}]},
      "portfolio": [{"id": "bad_type", "type": "straddle", "underlying": "A",
                     "strike": 100.0, "maturity": 1.0}]})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 100.0, "volatility": 0.2}]},
      "portfolio": [{"type": "european_call", "underlying": "A",
                     "strike": 100.0, "maturity": 1.0}]})"));
}

QC_TEST(json_config, rejects_bad_correlation) {
    // Wrong shape
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}],
                 "correlation": [[1.0, 0.5], [0.5, 1.0]]},
      "portfolio": []})"));
    // Asymmetric
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2},
                            {"symbol": "B", "spot": 1.0, "volatility": 0.2}],
                 "correlation": [[1.0, 0.3], [0.7, 1.0]]},
      "portfolio": []})"));
    // Symmetric, unit-diagonal, in-range, but not positive definite.
    QC_CHECK_THROWS(load_engine_input(R"({
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2},
                            {"symbol": "B", "spot": 1.0, "volatility": 0.2},
                            {"symbol": "C", "spot": 1.0, "volatility": 0.2}],
                 "correlation": [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9],
                                 [0.9, -0.9, 1.0]]},
      "portfolio": []})"));
}

QC_TEST(json_config, rejects_oversized_workloads) {
    QC_CHECK_THROWS(load_engine_input(R"({
      "simulation": {"paths": 99999999999},
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}]},
      "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "simulation": {"paths": 100000001},
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}]},
      "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "risk": {"scenarios": 10000001},
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}]},
      "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "simulation": {"time_steps": 100001},
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}]},
      "portfolio": []})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "simulation": {"paths": 100000000},
      "market": {"risk_free_rate": 0.05,
                 "assets": [{"symbol": "A", "spot": 1.0, "volatility": 0.2}]},
      "portfolio": [{"id":"A1","type":"asian_call","underlying":"A",
                     "strike":1.0,"maturity":1.0,"time_steps":51}]})"));
    QC_CHECK_THROWS(load_engine_input(R"({
      "risk": {"scenarios": 10000000},
      "market": {"risk_free_rate": 0.05, "assets": [
        {"symbol":"A","spot":1,"volatility":0.2},
        {"symbol":"B","spot":1,"volatility":0.2},
        {"symbol":"C","spot":1,"volatility":0.2},
        {"symbol":"D","spot":1,"volatility":0.2},
        {"symbol":"E","spot":1,"volatility":0.2},
        {"symbol":"F","spot":1,"volatility":0.2}]},
      "portfolio": []})"));

    // Unknown fields are rejected at every object level instead of silently
    // accepting likely misspellings.
    QC_CHECK_THROWS(load_engine_input(R"({"extra":1,"market":{},"portfolio":[]})"));
    QC_CHECK_THROWS(load_engine_input(R"({"simulation":{"pathz":1},"market":{},"portfolio":[]})"));
    QC_CHECK_THROWS(load_engine_input(R"({"market":{"risk_free_rate":0.05,"assets":[],"extra":1},"portfolio":[]})"));
    QC_CHECK_THROWS(load_engine_input(R"({"market":{"risk_free_rate":0.05,"assets":[{"symbol":"A","spot":1,"volatility":0.2,"extra":1}]},"portfolio":[]})"));
    QC_CHECK_THROWS(load_engine_input(R"({"market":{"risk_free_rate":0.05,"assets":[{"symbol":"A","spot":1,"volatility":0.2}]},"portfolio":[{"id":"C","type":"call","underlying":"A","strike":1,"maturity":1,"extra":1}]})"));
    QC_CHECK_THROWS(load_engine_input(R"({"market":{"risk_free_rate":0.05,"assets":[{"symbol":"A","spot":1,"volatility":0.2}]},"portfolio":[],"risk":{"levelz":[0.95]}})"));
}
