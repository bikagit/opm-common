/*
  Copyright (c) 2020 Equinor ASA

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

#include <opm/output/eclipse/WriteRPT.hpp>

#include <algorithm>
#include <functional>

#include <opm/parser/eclipse/EclipseState/Schedule/Schedule.hpp>
#include <opm/parser/eclipse/EclipseState/Grid/EclipseGrid.hpp>

namespace {

    constexpr char field_separator   {  ':' } ;
    constexpr char field_padding     {  ' ' } ;
    constexpr char record_separator  { '\n' } ;
    constexpr char section_separator { '\n' } ;
    constexpr char divider_character {  '-' } ;

    void left_align(std::string& string, std::size_t width, std::size_t = 0) {
        if (string.size() < width) {
            string.append(std::string(width - string.size(), field_padding));
        }
    }

    void left_header(std::string& string, std::size_t width, std::size_t line_number) {
        if (line_number == 0) {
            left_align(string, width, line_number);
        } else {
            string.clear();
        }
    }

    void right_align(std::string& string, std::size_t width, std::size_t = 0) {
        if (string.size() < width) {
            string = std::string(width - string.size(), field_padding) + string;
        }
    }

    void centre_align(std::string& string, std::size_t width, std::size_t = 0) {
        if (string.size() < width) {
            std::size_t extra_space { width - string.size() } ;
            std::size_t shift_one { extra_space % 2 } ;

            if (shift_one) {
                extra_space--;
            }

            std::size_t left { shift_one + extra_space / 2 }, right { extra_space / 2 } ;

            string = std::string(left, field_padding) + string + std::string(right, field_padding);
        }
    }

    struct context {
        const Opm::Schedule& sched;
        const Opm::EclipseGrid& grid;
    };

    template<typename T>
    const std::string& unimplemented(const T&, const context&, std::size_t, std::size_t) {
        static const std::string s { } ;

        return s;
    }

    template<typename T, std::size_t header_height>
    struct column {
        using fetch_function = std::function<std::string(const T&, const context&, std::size_t, std::size_t)>;
        using format_function = std::function<void(std::string&, std::size_t, std::size_t)>;

        std::size_t internal_width;
        std::array<std::string, header_height> header;

        fetch_function fetch = unimplemented<T>;
        format_function format = centre_align;

        void print(std::ostream& os, const T& data, const context& ctx, std::size_t sub_report, std::size_t line_number) const {
            std::string string_data { fetch(data, ctx, sub_report, line_number) } ;
            format(string_data, internal_width, line_number);
            centre_align(string_data, total_width());
            os << string_data;
        }

        void print_header(std::ostream& os, std::size_t row) const {
            std::string header_line { header[row] } ;
            centre_align(header_line, total_width());
            os << header_line;
        }

        constexpr std::size_t total_width() const {
            return internal_width + 2;
        }
    };

    template<typename T, std::size_t header_height>
    struct table: std::vector<column<T, header_height>> {
        using std::vector<column<T, header_height>>::vector;

        std::size_t total_width() const {
            std::size_t r { 1 + this->size() } ;

            for (const auto& column : *this) {
                r += column.total_width();
            }

            return r;
        }

        void print_divider(std::ostream& os, char padding = divider_character) const {
            os << std::string(total_width(), padding) << record_separator;
        }

        void print_header(std::ostream& os) const {
            print_divider(os);
            for (size_t i { 0 }; i < header_height; ++i) {
                for (const auto& column : *this) {
                    os << field_separator;

                    column.print_header(os, i);
                }

                os << field_separator << record_separator;
            }
            print_divider(os);
        }

        void print_data(std::ostream& os, const std::vector<T>& lines, const context& ctx, std::size_t sub_report) const {
            std::size_t line_number = 0;
            for (const auto& line : lines) {

                for (const auto& column : *this) {
                    os << field_separator;
                    column.print(os, line, ctx, sub_report, line_number);
                }

                os << field_separator << record_separator;

                ++line_number;
            }
        }
    };


    template<typename InputType, typename OutputType, std::size_t header_height>
    struct report {

        std::string title;
        std::string decor;
        table<OutputType, header_height> column_definition;
        const context ctx;

        report(const std::string& _title, const table<OutputType, header_height>& _coldef, const context& _ctx)
            : title              { _title           }
            , decor              { underline(title) }
            , column_definition  { _coldef          }
            , ctx                { _ctx             }
        {
            centre_align(title, column_definition.total_width());
            centre_align(decor, column_definition.total_width());
        }

        std::string underline(const std::string& string) const {
            return std::string(string.size(), divider_character);
        }

        void print_header(std::ostream& os) const {
            os << title << record_separator;
            os << decor << record_separator;
            os << section_separator;
            column_definition.print_header(os);
        }

        void print_data(std::ostream& os, const std::vector<OutputType>& data, std::size_t sub_report, char bottom_border = '-') const {
            column_definition.print_data(os, data, this->ctx, sub_report);
            column_definition.print_divider(os, bottom_border);
        }

        void print_footer(std::ostream& os, const std::vector<std::pair<int, std::string>>& footnotes) const {
            for (const auto& fnote: footnotes)
                os << fnote.first << ": " << fnote.second << std::endl;
            os << std::endl << std::endl;
        }
    };
}

namespace {

    inline std::string box_line(const std::pair<std::string,std::string>& textp, std::size_t line) {
        if (line == 1 || line == 2) {
            std::string text { line == 1 ? textp.first : textp.second } ;
            left_align(text, 72);

            return "*" + text + "*";
        } else {
            return std::string(74, '*');
        }
    }

    std::string wrap_string_for_header(const std::string& string) {
        std::string r { string } ;
        left_align(r, 27);
        centre_align(r, 29);

        return r;
    }

    const std::string header_days_string { "WELSPECS AT       0.00 DAYS" } ;
    std::string header_days(const Opm::Schedule& , std::size_t ) {
        return wrap_string_for_header(header_days_string); // TODO: Calculate properly
    }

    const std::string report_line_string { "REPORT   0     31 DEC 2007"  } ;
    std::string report_line(const Opm::Schedule&, std::size_t ) {
        return wrap_string_for_header(report_line_string); // TODO: Calculate properly
    }

    const std::string header_version_string { "FLOW" } ;
    std::string version_string() {
        return wrap_string_for_header(header_version_string); // TODO: Include in build setup and fetch
    }

    const std::string header_run_time_string { "RUN AT 12:41 ON 12 SEP 2016" } ;
    std::string run_time() {
        return wrap_string_for_header(header_run_time_string); // TODO: Calculate properly
    }


    void write_report_header(std::ostream& os, const Opm::Schedule& schedule, std::size_t report_step) {
        const static std::string filler { std::string(29, ' ') } ;

        const std::pair<std::string,std::string> box_text { "", "" } ;
        os <<
            filler                             << box_line(box_text, 0) << filler           << record_separator <<
            header_days(schedule, report_step) << box_line(box_text, 1) << version_string() << record_separator <<
            report_line(schedule, report_step) << box_line(box_text, 2) << run_time()       << record_separator <<
            filler                             << box_line(box_text, 3) << filler           << record_separator <<
            section_separator;
    }

}

namespace {

    struct WellWrapper {
        const Opm::Well& well;

        WellWrapper(const Opm::Well& well_arg) :
            well(well_arg)
        { }

        std::string well_name(const context&, std::size_t, std::size_t) const {
            return well.name();
        }

        std::string group_name(const context&, std::size_t, std::size_t) const {
            return well.groupName();
        }

        std::string wellhead_location(const context&, std::size_t, std::size_t) const {
            auto i { std::to_string(well.getHeadI() + 1) }, j { std::to_string(well.getHeadJ() + 1) } ;

            right_align(i, 3);
            right_align(j, 3);

            return i + ", " + j;
        }

        std::string reference_depth(const context&, std::size_t, std::size_t) const {
            return std::to_string(well.getRefDepth()).substr(0,6);
        }

        std::string preferred_phase(const context&, std::size_t, std::size_t) const {
            std::ostringstream ss;

            ss << well.getPreferredPhase();

            return ss.str().substr(0, 3);
        }

        std::string pvt_tab(const context&, std::size_t, std::size_t) const {
            return std::to_string( well.pvt_table_number() );
        }

        std::string shut_status(const context&, std::size_t, std::size_t) const {
            return Opm::Well::Status2String(well.getStatus());
        }

        std::string region_number(const context&, std::size_t, std::size_t) const {
            return std::to_string( well.fip_region_number() );
        }

        std::string dens_calc(const context&, std::size_t, std::size_t) const {
            if (well.segmented_density_calculation())
                return "SEG";
            return "AVG";
        }

        /*
          Don't know what the D-FACTOR represents, but all examples just show 0;
          we have therefor hardcoded that for now.
        */
        std::string D_factor(const context&, std::size_t, std::size_t) const {
            return "0";
        }

        std::string cross_flow(const context&, std::size_t, std::size_t) const {
            return well.getAllowCrossFlow() ? "YES" : "NO";
        }

        std::string drainage_radius(const context&, std::size_t, std::size_t) const {
            if (well.getDrainageRadius() == 0)
                return "P.EQUIV.R";
            return std::to_string(well.getDrainageRadius()).substr(0,6);
        }

        std::string gas_inflow(const context&, std::size_t, std::size_t) const {
            return Opm::Well::GasInflowEquation2String( well.gas_inflow_equation() );
        }
    };

    const table<WellWrapper, 3> well_specification_table {
       {  8, { "WELL"       , "NAME"       ,               }, &WellWrapper::well_name        , left_align  },
       {  8, { "GROUP"      , "NAME"       ,               }, &WellWrapper::group_name       , left_align  },
       {  8, { "WELLHEAD"   , "LOCATION"   , "( I, J )"    }, &WellWrapper::wellhead_location, left_align  },
       {  8, { "B.H.REF"    , "DEPTH"      , "METRES"      }, &WellWrapper::reference_depth  , right_align },
       {  5, { "PREF-"      , "ERRED"      , "PHASE"       }, &WellWrapper::preferred_phase  ,             },
       {  8, { "DRAINAGE"   , "RADIUS"     , "METRES"      }, &WellWrapper::drainage_radius  ,             },
       {  4, { "GAS"        , "INFL"       , "EQUN"        }, &WellWrapper::gas_inflow       ,             },
       {  7, { "SHUT-IN"    , "INSTRCT"    ,               }, &WellWrapper::shut_status      ,             },
       {  5, { "CROSS"      , "FLOW"       , "ABLTY"       }, &WellWrapper::cross_flow       ,             },
       {  3, { "PVT"        , "TAB"        ,               }, &WellWrapper::pvt_tab          ,             },
       {  4, { "WELL"       , "DENS"       , "CALC"        }, &WellWrapper::dens_calc        ,             },
       {  3, { "FIP"        , "REG"        ,               }, &WellWrapper::region_number    ,             },
       { 11, { "WELL"       , "D-FACTOR 1" , "DAY/SM3"     }, &WellWrapper::D_factor         ,             }};



void report_well_specification_data(std::ostream& os, const std::vector<Opm::Well>& data, const context& ctx) {
    report<Opm::Well, WellWrapper, 3> well_specification { "WELL SPECIFICATION DATA", well_specification_table, ctx};
    std::vector<WellWrapper> wrapper_data;
    std::transform(data.begin(), data.end(), std::back_inserter(wrapper_data), [](const Opm::Well& well) { return WellWrapper(well); });

    well_specification.print_header(os);
    well_specification.print_data(os, wrapper_data, 0);
    well_specification.print_footer(os, {{1, "The WELL D-FACTOR is not implemented - and the report will always show the default value 0."}});
}

}

namespace {

    struct WellConnection {
        const Opm::Well& well;
        const Opm::Connection& connection;

        WellConnection(const Opm::Well& well_arg, const Opm::Connection& connection_arg) :
            well(well_arg),
            connection(connection_arg)
        {}


        const std::string& well_name(const context&, std::size_t, std::size_t) const {
            return well.name();
        }

        std::string grid_block(const context&, std::size_t, std::size_t) const {
            const std::array<int,3> ijk { connection.getI() + 1, connection.getJ() + 1, connection.getK() + 1 } ;

            auto compose_coordinates = [](std::string& out, int in) -> std::string {
                constexpr auto delimiter { ',' } ;
                std::string coordinate_part { std::to_string(in) } ;
                right_align(coordinate_part, 3);

                return out.empty()
                    ? coordinate_part
                    : out + delimiter + coordinate_part;
            };

            return std::accumulate(std::begin(ijk), std::end(ijk), std::string {}, compose_coordinates);
        }

        std::string cmpl_no(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.complnum());
        }

        std::string centre_depth(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.depth()).substr(0, 6);
        }

        std::string open_shut(const context&, std::size_t, std::size_t) const {
            return Opm::Connection::State2String(connection.state());
        }

        std::string sat_tab(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.satTableId());
        }

        std::string conn_factor(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.CF()).substr(0, 10);
        }

        std::string int_diam(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.rw() * 2).substr(0, 8);
        }

        std::string kh_value(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.Kh()).substr(0, 9);
        }

        std::string skin_factor(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.skinFactor()).substr(0, 8);
        }

        std::string sat_scaling(const context&, std::size_t, std::size_t) const {
            return "";
        }

        const std::string dfactor(const context&, std::size_t, std::size_t) const {
            return "0";
        }

    };

    const table<WellConnection, 3> connection_table {
       {  7, {"WELL"                   ,"NAME"                     ,                         }, &WellConnection::well_name       , left_align  },
       { 12, {"GRID"                   ,"BLOCK"                    ,                         }, &WellConnection::grid_block      ,             },
       {  3, {"CMPL"                   ,"NO#"                      ,                         }, &WellConnection::cmpl_no         , right_align },
       {  7, {"CENTRE"                 ,"DEPTH"                    ,"METRES"                 }, &WellConnection::centre_depth    , right_align },
       {  3, {"OPEN"                   ,"SHUT"                     ,                         }, &WellConnection::open_shut       ,             },
       {  3, {"SAT"                    ,"TAB"                      ,                         }, &WellConnection::sat_tab         ,             },
       {  8, {"CONNECTION"             ,"FACTOR*"                  ,"CPM3/D/B"               }, &WellConnection::conn_factor     , right_align },
       {  6, {"INT"                    ,"DIAM"                     ,"METRES"                 }, &WellConnection::int_diam        , right_align },
       {  7, {"K  H"                   ,"VALUE"                    ,"MD.METRE"               }, &WellConnection::kh_value        , right_align },
       {  6, {"SKIN"                   ,"FACTOR"                   ,                         }, &WellConnection::skin_factor     , right_align },
       { 10, {"CONNECTION"             ,"D-FACTOR 1"               ,"DAY/SM3"                }, &WellConnection::dfactor         ,             },
       { 23, {"SATURATION SCALING DATA","SWMIN SWMAX SGMIN SGMAX 2",                         }, &WellConnection::sat_scaling     ,             }};

}

namespace {

    struct SegmentConnection {
        const Opm::Well& well;
        const Opm::Connection& connection;
        const Opm::Segment& segment;


        SegmentConnection(const Opm::Well& well_arg, const Opm::Connection& conn_arg, const Opm::Segment& segment_arg) :
            well(well_arg),
            connection(conn_arg),
            segment(segment_arg)
        {}

        const std::string& well_name(const context&, std::size_t, std::size_t) const {
            return well.name();
        }

        std::string connection_grid(const context& ctx, std::size_t sub_report, std::size_t n) const {
            const WellConnection wc { well, connection } ;

            return wc.grid_block(ctx, sub_report, n);
        }

        std::string segment_number(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.segmentNumber());
        }

        std::string branch_id(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.branchNumber());
        }

        std::string length_end_segmt(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.totalLength()).substr(0, 6);
        }

        std::string connection_depth(const context&, std::size_t, std::size_t) const {
            return std::to_string(connection.depth()).substr(0, 6);
        }

        std::string segment_depth(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.depth()).substr(0, 6);
        }

        std::string grid_block_depth(const context& ctx, std::size_t, std::size_t) const {
            return std::to_string( ctx.grid.getCellDepth( connection.global_index() )).substr(0,6);
        }


        static void ws_format(std::string& string, std::size_t, std::size_t i) {
            if (i == 0) {
                left_align(string, 8, i);
            } else {
                right_align(string, 8, i);
            }
        }

    };



    struct WellSegment {
        const Opm::Well& well;
        const Opm::Segment& segment;

        WellSegment(const Opm::Well& well_arg, const Opm::Segment& segment_arg) :
            well(well_arg),
            segment(segment_arg)
        {}

        std::string well_name_seg(const context&, std::size_t sub_report, std::size_t n) const {
            if (sub_report > 0)
                return "";

            if (n == 0)
                return well.name();

            if (n == 1)
                return Opm::WellSegments::CompPressureDropToString(well.getSegments().compPressureDrop());

            return "";
        }

        std::string segment_number(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.segmentNumber());
        }

        std::string branch_number(const context&, std::size_t, std::size_t n) const {
            if (n == 0)
                return std::to_string(segment.branchNumber());
            return "";
        }

        std::string main_inlet(const context&, std::size_t, std::size_t) const {
            const auto& inlets { segment.inletSegments() } ;

            if (inlets.size() != 0) {
                return std::to_string(segment.inletSegments().front());
            } else {
                return "0";
            }
        }

        std::string outlet(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.outletSegment());
        }

        std::string total_length(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.totalLength()).substr(0, 6);
        }

        std::string length(const context& ctx, std::size_t sub_report, std::size_t line_number) const {
            if (segment.segmentNumber() == 1)
                return total_length(ctx, sub_report, line_number);

            const auto& segments = well.getSegments();
            const auto& outlet_segment = segments.getFromSegmentNumber( segment.outletSegment() );
            return std::to_string( segment.totalLength() - outlet_segment.totalLength() ).substr(0, 6);
        }

        std::string t_v_depth(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.depth()).substr(0, 6);
        }

        std::string depth_change(const context& ctx, std::size_t sub_report, std::size_t line_number) const {
            if (segment.segmentNumber() == 1)
                return t_v_depth(ctx, sub_report, line_number);

            const auto& segments = well.getSegments();
            const auto& outlet_segment = segments.getFromSegmentNumber( segment.outletSegment() );
            return std::to_string( segment.depth() - outlet_segment.depth() ).substr(0, 6);
        }

        std::string internal_diameter(const context&, std::size_t, std::size_t) const {
            const auto number { segment.internalDiameter() } ;

            if (number != Opm::Segment::invalidValue()) {
                return std::to_string(number).substr(0, 6);
            } else {
                return "0";
            }
        }

        std::string roughness(const context&, std::size_t, std::size_t) const {
            const auto number { segment.roughness() } ;

            if (number != Opm::Segment::invalidValue()) {
                return std::to_string(number).substr(0, 8);
            } else {
                return "0";
            }
        }

        std::string cross_section(const context&, std::size_t, std::size_t) const {
            const auto number { segment.crossArea() } ;

            if (number != Opm::Segment::invalidValue()) {
                return std::to_string(number).substr(0, 7);
            } else {
                return "0";
            }
        }

        std::string volume(const context&, std::size_t, std::size_t) const {
            return std::to_string(segment.volume()).substr(0, 5);
        }

        std::string pressure_drop_mult(const context&, std::size_t, std::size_t) const {
            return std::to_string(1.0).substr(0, 5);
        }


        static void ws_format(std::string& string, std::size_t, std::size_t i) {
            if (i == 0) {
                left_align(string, 8, i);
            } else {
                right_align(string, 8, i);
            }
        }

    };


    const table<SegmentConnection, 3> msw_connection_table = {
        {  8, {"WELL"       , "NAME"       ,              }, &SegmentConnection::well_name        , left_header },
        {  9, {"CONNECTION" , ""           ,              }, &SegmentConnection::connection_grid  ,             },
        {  5, {"SEGMENT"    , "NUMBER"     ,              }, &SegmentConnection::segment_number   , right_align },
        {  8, {"BRANCH"     , "ID"         ,              }, &SegmentConnection::branch_id        ,             },
        {  9, {"TUB LENGTH" , "START PERFS", "METRES"     }, unimplemented<SegmentConnection>     , right_align },
        {  9, {"TUB LENGTH" , "END PERFS"  , "METRES"     }, unimplemented<SegmentConnection>     , right_align },
        {  9, {"TUB LENGTH" , "CENTR PERFS", "METRES"     }, unimplemented<SegmentConnection>     , right_align },
        {  9, {"TUB LENGTH" , "END SEGMT"  , "METRES"     }, &SegmentConnection::length_end_segmt , right_align },
        {  8, {"CONNECTION" , "DEPTH"      , "METRES"     }, &SegmentConnection::connection_depth , right_align },
        {  8, {"SEGMENT"    , "DEPTH"      , "METRES"     }, &SegmentConnection::segment_depth    , right_align },
        {  9, {"GRID BLOCK" , "DEPTH"      , "METRES"     }, &SegmentConnection::grid_block_depth , right_align },
    };

    const table<WellSegment, 3> msw_well_table = {
        {  6, { "WELLNAME"  , "AND"        , "SEG TYPE"   }, &WellSegment::well_name_seg       , &WellSegment::ws_format },
        {  3, { "SEG"       , "NO"         , ""           }, &WellSegment::segment_number      , right_align             },
        {  3, { "BRN"       , "NO"         , ""           }, &WellSegment::branch_number       , right_align             },
        {  5, { "MAIN"      , "INLET"      , "SEGMENT"    }, &WellSegment::main_inlet          , right_align             },
        {  5, { ""          , "OUTLET"     , "SEGMENT"    }, &WellSegment::outlet              , right_align             },
        {  7, { "SEGMENT"   , "LENGTH"     , "METRES"     }, &WellSegment::length              , right_align             },
        {  8, { "TOT LENGTH", "TO END"     , "METRES"     }, &WellSegment::total_length        , right_align             },
        {  8, { "DEPTH"     , "CHANGE"     , "METRES"     }, &WellSegment::depth_change        , right_align             },
        {  8, { "T.V. DEPTH", "AT END"     , "METRES"     }, &WellSegment::t_v_depth           , right_align             },
        {  6, { "DIA OR F"  , "SCALING"    , "METRES"     }, &WellSegment::internal_diameter   , right_align             },
        {  8, { "VFP TAB OR", "ABS ROUGHN" , "METRES"     }, &WellSegment::roughness           , right_align             },
        {  7, { "AREA"      , "X-SECTN"    , "M**2"       }, &WellSegment::cross_section       , right_align             },
        {  7, { "VOLUME"    , ""           , "M3"         }, &WellSegment::volume              , right_align             },
        {  8, { "P DROP"    , "MULT"       , "FACTOR 1"   }, &WellSegment::pressure_drop_mult  , right_align             },
    };
}

namespace {

void report_well_connection_data(std::ostream& os, const std::vector<Opm::Well>& data, const context& ctx) {
    const report<Opm::Well, WellConnection, 3> well_connection { "WELL CONNECTION DATA", connection_table, ctx};
    well_connection.print_header(os);

    std::size_t sub_report = 0;
    for (const auto& well : data) {
        std::vector<WellConnection> wrapper_data;
        const auto& connections = well.getConnections();
        std::transform(connections.begin(), connections.end(), std::back_inserter(wrapper_data), [&well]( const Opm::Connection& connection) { return WellConnection(well, connection); });

        well_connection.print_data(os, wrapper_data, sub_report);
        sub_report++;
    }
    well_connection.print_footer(os, {{1, "The well connection D-FACTOR is not implemented in opm and the report will always show 0."},
                                      {2, "The saturation scaling data has not been implemented in the report and will always be blank."}});
}

}

void Opm::RptIO::workers::write_WELSPECS(std::ostream& os, unsigned, const Opm::Schedule& schedule, const Opm::EclipseGrid& grid, std::size_t report_step) {
    auto well_names = schedule.changed_wells(report_step);
    if (well_names.empty())
        return;

    context ctx{schedule, grid};
    std::vector<Well> changed_wells;
    std::transform(well_names.begin(), well_names.end(), std::back_inserter(changed_wells), [&report_step, &schedule](const std::string& wname) { return schedule.getWell(wname, report_step); });

    write_report_header(os, schedule, report_step);
    report_well_specification_data(os, changed_wells, ctx);
    report_well_connection_data(os, changed_wells, ctx);

    for (const auto& well : changed_wells) {
        if (well.isMultiSegment()) {
            {
                const report<Opm::Well, WellSegment, 3> msw_data { "MULTI-SEGMENT WELL: SEGMENT STRUCTURE", msw_well_table, ctx};
                msw_data.print_header(os);
                std::size_t sub_report = 0;
                const auto& segments = well.getSegments();
                for (const auto& branch : segments.branches()) {
                    std::vector<WellSegment> wrapper_data;
                    const auto& branch_segments = segments.branchSegments(branch);
                    std::transform(branch_segments.begin(), branch_segments.end(), std::back_inserter(wrapper_data), [&well](const Opm::Segment& segment) { return WellSegment(well, segment); });

                    sub_report++;
                    if (sub_report == (segments.branches().size()))
                        msw_data.print_data(os, wrapper_data, sub_report - 1, '=');
                    else
                        msw_data.print_data(os, wrapper_data, sub_report - 1, '-');
                }
                msw_data.print_footer(os, {{1, "The pressure drop multiplier is not implemented in opm/flow and will always show the default value 1.0."}});
            }
            {
                const report<Opm::Well, SegmentConnection, 3> msw_connection { "MULTI-SEGMENT WELL: CONNECTION DATA", msw_connection_table,  ctx};
                msw_connection.print_header(os);
                {
                    std::vector<SegmentConnection> wrapper_data;
                    const auto& connections = well.getConnections();
                    const auto& segments = well.getSegments();
                    std::transform(connections.begin(), connections.end(), std::back_inserter(wrapper_data),
                                   [&well, &segments] (const Opm::Connection& connection) { return SegmentConnection(well, connection, segments.getFromSegmentNumber(connection.segment())); });
                    msw_connection.print_data(os, wrapper_data, 0, '=');
                }
                msw_connection.print_footer(os, {});
            }
        }
    }
}