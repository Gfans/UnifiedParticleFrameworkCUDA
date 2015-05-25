/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

namespace thrust
{

namespace system
{


error_code make_error_code(cuda::errc::errc_t e)
{
  return error_code(static_cast<int>(e), cuda_category());
} // end make_error_code()


error_condition make_error_condition(cuda::errc::errc_t e)
{
  return error_condition(static_cast<int>(e), cuda_category());
} // end make_error_condition()


namespace cuda
{

namespace detail
{


class cuda_error_category
  : public error_category
{
  public:
    inline cuda_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "cuda";
    }

    inline virtual std::string message(int ev) const
    {
      static const std::string unknown_err("Unknown error");
      const char *c_str = ::cudaGetErrorString(static_cast<cudaError_t>(ev));
      return c_str ? std::string(c_str) : unknown_err;
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace cuda::errc;

      if(ev < ::cudaErrorApiFailureBase)
      {
        return make_error_condition(static_cast<errc_t>(ev));
      }

      return system_category().default_error_condition(ev);
    }
}; // end cuda_error_category

} // end detail

} // end namespace cuda


const error_category &cuda_category(void)
{
  static const cuda::detail::cuda_error_category result;
  return result;
}


} // end namespace system

} // end namespace thrust

