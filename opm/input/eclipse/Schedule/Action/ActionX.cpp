/*
  Copyright 2018 Statoil ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <opm/input/eclipse/Schedule/Action/ActionX.hpp>

#include <opm/common/utility/OpmInputError.hpp>

#include <opm/io/eclipse/rst/action.hpp>

#include <opm/input/eclipse/Schedule/Action/ActionValue.hpp>
#include <opm/input/eclipse/Schedule/Action/Actdims.hpp>
#include <opm/input/eclipse/Schedule/Action/State.hpp>
#include <opm/input/eclipse/Schedule/Well/WellMatcher.hpp>

#include <opm/input/eclipse/Utility/Typetools.hpp>

#include <opm/input/eclipse/Deck/DeckKeyword.hpp>
#include <opm/input/eclipse/Deck/DeckOutput.hpp>

#include <opm/input/eclipse/Parser/ParseContext.hpp>
#include <opm/input/eclipse/Parser/ParserKeywords/W.hpp>

#include <ctime>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ActionParser.hpp"

namespace Opm {
namespace Action {
namespace {

std::string dequote(const std::string& token, const std::optional<KeywordLocation>& location) {
    if (token[0] == '\'') {
        if (token.back() == '\'')
            return token.substr(1, token.size() - 2);
        else {
            auto msg = fmt::format("Unbalanced quote for token: {}", token);
            if (location.has_value())
                throw OpmInputError(msg, location.value());
            else
                throw std::logic_error(msg);
        }
    } else
        return token;
}

}


bool ActionX::valid_keyword(const std::string& keyword)
{
    static const auto actionx_allowed_list = std::unordered_set<std::string> {
        "BOX",

        "COMPLUMP", "COMPDAT", "COMPSEGS",

        "ENDBOX", "EXIT",

        "GCONINJE", "GCONPROD", "GCONSUMP",
        "GLIFTOPT",
        "GRUPNET", "GRUPTARG", "GRUPTREE",
        "GSATINJE", "GSATPROD",

        "MULTX", "MULTX-", "MULTY", "MULTY-", "MULTZ", "MULTZ-",
        "NEXT", "NEXTSTEP",

        "UDQ",

        "WCONINJE", "WCONPROD",
        "WECON", "WEFAC",
        "WELOPEN", "WELPI", "WELSEGS", "WELSPECS", "WELTARG",
        "WGRUPCON",
        "WLIST",
        "WPIMULT",
        "WSEGVALV",
        "WTEST", "WTMULT",

        "TEST",
    };

    return actionx_allowed_list.find(keyword) != actionx_allowed_list.end();
}

std::tuple<ActionX, std::vector<std::pair<std::string, std::string>>>
parseActionX(const DeckKeyword& kw, const Actdims& actdims,
             std::time_t start_time)
{
    std::vector<std::pair<std::string, std::string>> condition_errors;
    std::vector<std::string> tokens;
    std::vector<Condition> conditions;
    auto record = kw.getRecord(0);
    const std::string name = record.getItem("NAME").getTrimmedString(0);

    for (size_t record_index = 1; record_index < kw.size(); record_index++) {
        const auto& cond_tokens = RawString::strings( kw.getRecord(record_index)
                                                      .getItem("CONDITION").getData<RawString>() );

        std::transform(cond_tokens.begin(), cond_tokens.end(),
                       std::back_inserter(tokens),
                       [&kw](const auto& token) { return dequote(token, kw.location()); });

        conditions.emplace_back(cond_tokens, kw.location());
    }
    if (conditions.empty())
        condition_errors.push_back({ParseContext::ACTIONX_NO_CONDITION,
                fmt::format("Action {} is missing a condition.", name)});

    if (conditions.size() > actdims.max_conditions())
        condition_errors.push_back({ ParseContext::ACTIONX_CONDITION_ERROR,
                fmt::format("Action {} has too many conditions - adjust item "
                            "4 of ACTDIMS to at least {}",
                            name, conditions.size())});

    try
    {
        return { ActionX(name,
                         record.getItem("NUM").get<int>(0),
                         record.getItem("MIN_WAIT").getSIDouble(0),
                         start_time, std::move(conditions), std::move(tokens)),
            condition_errors};
    }
    catch(const std::invalid_argument& e)
    {
        condition_errors.push_back({ ParseContext::ACTIONX_CONDITION_ERROR,
                fmt::format("condition of action {} has the following error: {}", name, e.what())});
        return {ActionX(kw.getRecord(0), start_time), condition_errors};
    }
}

ActionX::ActionX() :
    m_start_time(0)
{
}


ActionX::ActionX(const std::string& name, size_t max_run, double min_wait, std::time_t start_time) :
    m_name(name),
    m_max_run(max_run),
    m_min_wait(min_wait),
    m_start_time(start_time)
{}

ActionX::ActionX(const RestartIO::RstAction& rst_action)
    : m_name(rst_action.name)
    , m_max_run(rst_action.max_run)
    , m_min_wait(rst_action.min_wait),
      m_start_time(rst_action.start_time)
{
    std::vector<std::string> tokens;
    for (const auto& rst_condition : rst_action.conditions) {
        this->m_conditions.emplace_back(rst_condition);

        const auto rst_tokens = rst_condition.tokens();
        std::transform(rst_tokens.begin(), rst_tokens.end(),
                       std::back_inserter(tokens),
                       [](const auto& token) { return dequote(token, {}); });
    }
    this->condition = Action::AST(tokens);
    for (const auto& keyword : rst_action.keywords)
        this->addKeyword(keyword);
}



ActionX::ActionX(const DeckRecord& record, std::time_t start_time) :
    ActionX( record.getItem("NAME").getTrimmedString(0),
             record.getItem("NUM").get<int>(0),
             record.getItem("MIN_WAIT").getSIDouble(0),
             start_time )

{}





ActionX::ActionX(const std::string& name, size_t max_run, double min_wait,
                 std::time_t start_time,
                 const std::vector<Condition>&& conditions,
                 const std::vector<std::string>&& tokens)
    : m_name(name), m_max_run(max_run), m_min_wait(min_wait),
      m_start_time(start_time), condition(tokens), m_conditions(conditions)
{}


ActionX ActionX::serializationTestObject()
{
    ActionX result;
    result.m_name = "test";
    result.m_max_run = 1;
    result.m_min_wait = 2;
    result.m_start_time = 3;
    result.keywords = {DeckKeyword::serializationTestObject()};
    result.condition = Action::AST::serializationTestObject();
    Quantity quant;
    quant.quantity = "test1";
    quant.args = {"test2"};
    Condition cond;
    cond.lhs = quant;
    cond.rhs = quant;
    cond.logic = Logical::AND;
    cond.cmp = Comparator::GREATER_EQUAL;
    cond.cmp_string = "test3";
    result.m_conditions = {cond};

    return result;
}


void ActionX::addKeyword(const DeckKeyword& kw) {
    this->keywords.push_back(kw);
}



Action::Result ActionX::eval(const Action::Context& context) const {
    return this->condition.eval(context);
}


bool ActionX::ready(const State& state, std::time_t sim_time) const {
    auto run_count = state.run_count(*this);
    if (run_count >= this->max_run())
        return false;

    if (sim_time < this->start_time())
        return false;

    if (run_count == 0)
        return true;

    if (this->min_wait() <= 0)
        return true;

    auto last_run = state.run_time(*this);
    return std::difftime(sim_time, last_run) >= this->min_wait();
}


std::vector<DeckKeyword>::const_iterator ActionX::begin() const {
    return this->keywords.begin();
}

std::vector<DeckKeyword>::const_iterator ActionX::end() const {
    return this->keywords.end();
}


std::vector<std::string> ActionX::keyword_strings() const {
    std::vector<std::string> keyword_strings;
    std::string keyword_string;
    {
        std::stringstream ss;
        DeckOutput::format fmt;
        for (const auto& kw : this->keywords) {
            ss << kw;
            ss << fmt.keyword_sep;
        }

        keyword_string = ss.str();
    }

    std::size_t offset = 0;
    while (true) {
        auto eol_pos = keyword_string.find('\n', offset);
        if (eol_pos == std::string::npos)
            break;

        if (eol_pos > offset)
            keyword_strings.push_back(keyword_string.substr(offset, eol_pos - offset));

        offset = eol_pos + 1;
    }
    keyword_strings.push_back("ENDACTIO");

    return keyword_strings;
}


const std::vector<Condition>& ActionX::conditions() const {
    return this->m_conditions;
}

std::size_t ActionX::id() const {
    return this->m_id;
}

void ActionX::update_id(std::size_t id) {
    this->m_id = id;
}


bool ActionX::operator==(const ActionX& data) const {
    return this->name() == data.name() &&
           this->max_run() == data.max_run() &&
           this->min_wait() == data.min_wait() &&
           this->start_time() == data.start_time() &&
           this->id() == data.id() &&
           this->keywords == data.keywords &&
           this->condition == data.condition &&
           this->conditions() == data.conditions();
}


void ActionX::required_summary(std::unordered_set<std::string>& required_summary) const {
    this->condition.required_summary(required_summary);
}

std::vector<std::string>
ActionX::wellpi_wells(const WellMatcher& well_matcher,
                      const std::vector<std::string>& matching_wells) const
{
    auto wells = std::vector<std::string>{};

    for (const auto& kw : this->keywords) {
        if (kw.name() != ParserKeywords::WELPI::keywordName) {
            continue;
        }

        for (const auto& record : kw) {
            const auto wname_arg = record
                .getItem<ParserKeywords::WELPI::WELL_NAME>()
                .getTrimmedString(0);

            if (wname_arg == "?") {
                wells.insert(wells.end(), matching_wells.begin(), matching_wells.end());
            }
            else {
                const auto& well_range = well_matcher.wells(wname_arg);
                wells.insert(wells.end(), well_range.begin(), well_range.end());
            }
        }
    }

    if (! wells.empty()) {
        wells = well_matcher.sort(wells);

        // Note: std::unique() is sufficient here.  We don't need to care
        // about the particular sort order to identify duplicate strings,
        // only that any duplicates appear consecutively in the sequence.
        wells.erase(std::unique(wells.begin(), wells.end()), wells.end());
    }

    return wells;
}

}
}
