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
 *        multi-phase, multi-component fluid system assuming
 *        thermodynamic equilibrium.
 */
#ifndef OPM_COMPOSITIONAL_FLUID_STATE_HPP
#define OPM_COMPOSITIONAL_FLUID_STATE_HPP

#include <opm/common/utility/gpuDecorators.hpp>

#include <opm/material/fluidstates/FluidStateCompositionModules.hpp>
#include <opm/material/fluidstates/FluidStateDensityModules.hpp>
#include <opm/material/fluidstates/FluidStateEnthalpyModules.hpp>
#include <opm/material/fluidstates/FluidStateFugacityModules.hpp>
#include <opm/material/fluidstates/FluidStatePressureModules.hpp>
#include <opm/material/fluidstates/FluidStateSaturationModules.hpp>
#include <opm/material/fluidstates/FluidStateTemperatureModules.hpp>
#include <opm/material/fluidstates/FluidStateViscosityModules.hpp>
#include <opm/material/fluidstates/ModularFluidState.hpp>

namespace Opm {

/*!
 * \brief Represents all relevant thermodynamic quantities of a
 *        multi-phase, multi-component fluid system assuming
 *        thermodynamic equilibrium.
 */
template <class ValueT, class FluidSystem, bool storeEnthalpy=true>
class CompositionalFluidState;

// specialization for the enthalpy enabled case
template <class ValueT, class FluidSystem>
class CompositionalFluidState<ValueT, FluidSystem, true>
    : public ModularFluidState<ValueT,
                               FluidSystem::numPhases,
                               FluidSystem::numComponents,
                               FluidStateExplicitPressureModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateEquilibriumTemperatureModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitCompositionModule<ValueT, FluidSystem, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitFugacityModule<ValueT, FluidSystem::numPhases, FluidSystem::numComponents, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitSaturationModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitDensityModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitViscosityModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> >,
                               FluidStateExplicitEnthalpyModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, true> > >
{
public:
    /*!
     * \brief Return the fluid system used by this fluid state.
     *
     * \note This is needed for GPU compatibility for now.
     */
    OPM_HOST_DEVICE const FluidSystem& fluidSystem() const
    {
      static FluidSystem instance;
      return instance;
    }
};

// specialization for the enthalpy disabled case
template <class ValueT, class FluidSystem>
class CompositionalFluidState<ValueT, FluidSystem, false>
    : public ModularFluidState<ValueT,
                               FluidSystem::numPhases,
                               FluidSystem::numComponents,
                               FluidStateExplicitPressureModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateEquilibriumTemperatureModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateExplicitCompositionModule<ValueT, FluidSystem, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateExplicitFugacityModule<ValueT, FluidSystem::numPhases, FluidSystem::numComponents, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateExplicitSaturationModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateExplicitDensityModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateExplicitViscosityModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> >,
                               FluidStateNullEnthalpyModule<ValueT, FluidSystem::numPhases, CompositionalFluidState<ValueT, FluidSystem, false> > >
{
public:
    /*!
     * \brief Return the fluid system used by this fluid state.
     *
     * \note This is needed for GPU compatibility for now.
     */
    OPM_HOST_DEVICE const FluidSystem& fluidSystem() const
    {
      static FluidSystem instance;
      return instance;

    }
};

} // namespace Opm

#endif
