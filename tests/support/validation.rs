//! Default-on validation layer settings for library tests.

use {ash::vk, std::env};

pub(crate) const SKIP_SYNC: &str = "VK_GRAPH_SKIP_VALIDATION_SYNC";
pub(crate) const SKIP_SHADER_ACCESSES: &str = "VK_GRAPH_SKIP_VALIDATION_SHADER_ACCESSES_HEURISTIC";

/// Owns the BOOL32 values borrowed by the instance's layer-settings pNext chain.
pub(crate) struct ValidationSettings([vk::Bool32; 2]);

impl ValidationSettings {
    pub(crate) fn from_env() -> Self {
        Self::from_lookup(|name| env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self([SKIP_SYNC, SKIP_SHADER_ACCESSES].map(|name| {
            let skip = lookup(name).is_some_and(|value| {
                !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE")
            });
            vk::Bool32::from(!skip)
        }))
    }

    pub(crate) fn layer_settings(&self) -> [vk::LayerSettingEXT<'_>; 2] {
        std::array::from_fn(|index| {
            let name = [c"validate_sync", c"syncval_shader_accesses_heuristic"][index];
            let mut setting = vk::LayerSettingEXT::default()
                .layer_name(c"VK_LAYER_KHRONOS_validation")
                .setting_name(name)
                .ty(vk::LayerSettingTypeEXT::BOOL32);
            // ash's byte-slice setter counts bytes, but BOOL32 needs one aligned u32 value.
            setting.value_count = 1;
            setting.p_values = (&self.0[index] as *const vk::Bool32).cast();
            setting
        })
    }

    pub(crate) fn synchronization_enabled(&self) -> bool {
        self.0[0] == vk::TRUE
    }

    pub(crate) fn shader_accesses_enabled(&self) -> bool {
        self.synchronization_enabled() && self.0[1] == vk::TRUE
    }
}

#[cfg(test)]
mod test {
    use {super::*, std::ffi::CStr};

    #[test]
    fn settings_default_on_and_skip_independently() {
        for (sync, shader, expected) in [
            (None, None, [vk::TRUE, vk::TRUE]),
            (Some("1"), None, [vk::FALSE, vk::TRUE]),
            (None, Some("1"), [vk::TRUE, vk::FALSE]),
            (Some("true"), Some("1"), [vk::FALSE, vk::FALSE]),
            (Some("0"), Some("false"), [vk::TRUE, vk::TRUE]),
            (Some(""), Some("FALSE"), [vk::TRUE, vk::TRUE]),
        ] {
            let values = ValidationSettings::from_lookup(|name| {
                match name {
                    SKIP_SYNC => sync,
                    SKIP_SHADER_ACCESSES => shader,
                    _ => panic!("unexpected environment lookup: {name}"),
                }
                .map(str::to_owned)
            });
            for ((setting, expected), name) in values
                .layer_settings()
                .iter()
                .zip(expected)
                .zip([c"validate_sync", c"syncval_shader_accesses_heuristic"])
            {
                assert_eq!(setting.ty, vk::LayerSettingTypeEXT::BOOL32);
                assert_eq!(setting.value_count, 1);
                unsafe {
                    assert_eq!(
                        CStr::from_ptr(setting.p_layer_name),
                        c"VK_LAYER_KHRONOS_validation"
                    );
                    assert_eq!(CStr::from_ptr(setting.p_setting_name), name);
                    assert_eq!(*setting.p_values.cast::<vk::Bool32>(), expected);
                }
            }
        }
    }
}
