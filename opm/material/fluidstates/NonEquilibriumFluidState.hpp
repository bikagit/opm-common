// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief Represents all relevant thermodynamic quantities of a
 *        multi-phase, multi-component fluid system _not_ assuming
 *        thermodynamic equilibrium.
 */
#ifndef OPM_NON_EQUILIBRIUM_FLUID_STATE_HPP
#define OPM_NON_EQUILIBRIUM_FLUID_STATE_HPP

#include "ModularFluidState.hpp"

#include <opm/material/common/Valgrind.hpp>
#include <algorithm>

namespace Opm {

/*!
 * \brief Represents all relevant thermodynamic quantities of a
 *        multi-phase, multi-component fluid system _not_ assuming
 *        thermodynamic equilibrium.
 */
template <class ValueT, class FluidSystem, bool storeEnthalpy=true>
class NonEquilibriumFluidState;


// specialization for the enthalpy enabled case
template <class ValueType, class FluidSystem>
class NonEquilibriumFluidState<ValueType, FluidSystem, true>
    : public ModularFluidState<ValueType,
                               FluidSystem::numPhases,
                               FluidSystem::numComponents,
                               FluidStateExplicitPressureModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitTemperatureModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitCompositionModule<ValueType, FluidSystem, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitFugacityModule<ValueType, FluidSystem::numPhases, FluidSystem::numComponents, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitSaturationModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitDensityModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitViscosityModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> >,
                               FluidStateExplicitEnthalpyModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, true> > >
{
};

// specialization for the enthalpy disabled case
template <class ValueType, class FluidSystem>
class NonEquilibriumFluidState<ValueType, FluidSystem, false>
    : public ModularFluidState<ValueType,
                               FluidSystem::numPhases,
                               FluidSystem::numComponents,
                               FluidStateExplicitPressureModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitTemperatureModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitCompositionModule<ValueType, FluidSystem, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitFugacityModule<ValueType, FluidSystem::numPhases, FluidSystem::numComponents, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitSaturationModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitDensityModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateExplicitViscosityModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> >,
                               FluidStateNullEnthalpyModule<ValueType, FluidSystem::numPhases, NonEquilibriumFluidState<ValueType, FluidSystem, false> > >
{
};

} // namespace Opm

#endif
