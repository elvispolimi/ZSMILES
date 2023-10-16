#pragma once

#include <memory>
#include <string_view>

namespace smiles {
  enum class compressor_interface_type : std::uint_fast16_t { cpu = 0 };

  struct compressor_interface {
    virtual ~compressor_interface(void) {}

    virtual std::string_view operator()(const std::string_view&) = 0;
  };

  struct decompressor_interface {
    virtual ~decompressor_interface(void) {}

    virtual std::string_view operator()(const std::string_view&) = 0;
  };

  struct compressor_container {
    template<class T, class... Ts>
    inline void create(Ts&&... params) {
      implementation = std::make_unique<T>(std::forward<Ts>(params)...);
    }

    inline std::string_view operator()(const std::string_view& plain_description) {
      return implementation->operator()(plain_description);
    }

  private:
    std::unique_ptr<compressor_interface> implementation;
  };

  struct decompressor_container {
    template<class T, class... Ts>
    inline void create(Ts&&... params) {
      implementation = std::make_shared<T>(std::forward<Ts>(params)...);
    }

    inline std::string_view operator()(const std::string_view& compressed_description) {
      return implementation->operator()(compressed_description);
    }

  private:
    std::shared_ptr<decompressor_interface> implementation;
  };
} // namespace smiles
