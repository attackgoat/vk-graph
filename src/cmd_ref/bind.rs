use {
    super::{CommandRef, pipeline::PipelineCommandRef},
    crate::{
        ExecutionPipeline,
        driver::{compute::ComputePipeline, graphic::GraphicPipeline, ray_trace::RayTracePipeline},
    },
    std::marker::PhantomData,
};

/// A trait for pipelines which may be bound to a `CommandRef`.
///
/// See [`CommandRef::bind_pipeline`](super::cmd_ref::CommandRef::bind_pipeline) for details.
pub trait BindCommand<'a> {
    /// The resource reference type.
    type Ref;

    /// Binds the resource to a command.
    ///
    /// Returns a reference type.
    fn bind_cmd(self, _: CommandRef<'a>) -> Self::Ref;
}

macro_rules! bind_cmd_pipeline {
    ($name:ident) => {
        paste::paste! {
            impl<'a> BindCommand<'a> for &'a [<$name Pipeline>] {
                type Ref = PipelineCommandRef<'a, [<$name Pipeline>]>;

                fn bind_cmd(self, mut cmd: CommandRef<'a>) -> Self::Ref {
                    let cmd_ref = cmd.cmd_mut();
                    if cmd_ref.execs.last().unwrap().pipeline.is_some() {
                        // Binding from PipelineCommandRef -> PipelineCommandRef (changing shaders)
                        cmd_ref.execs.push(Default::default());
                    }

                    cmd_ref.execs.last_mut().unwrap().pipeline = Some(ExecutionPipeline::$name(self.clone()));

                    Self::Ref {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl<'a> BindCommand<'a> for [<$name Pipeline>] {
                type Ref = PipelineCommandRef<'a, [<$name Pipeline>]>;

                fn bind_cmd(self, mut cmd: CommandRef<'a>) -> Self::Ref {
                    let cmd_ref = cmd.cmd_mut();
                    if cmd_ref.execs.last().unwrap().pipeline.is_some() {
                        // Binding from PipelineCommandRef -> PipelineCommandRef (changing shaders)
                        cmd_ref.execs.push(Default::default());
                    }

                    cmd_ref.execs.last_mut().unwrap().pipeline = Some(ExecutionPipeline::$name(self));

                    Self::Ref {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl ExecutionPipeline {
                #[allow(unused)]
                pub(crate) fn [<is_ $name:snake>](&self) -> bool {
                    matches!(self, Self::$name(_))
                }

                #[allow(unused)]
                pub(crate) fn [<unwrap_ $name:snake>](&self) -> &[<$name Pipeline>] {
                    if let Self::$name(binding) = self {
                        &binding
                    } else {
                        panic!();
                    }
                }
            }
        }
    };
}

// Pipelines you can bind to a pass
bind_cmd_pipeline!(Compute);
bind_cmd_pipeline!(Graphic);
bind_cmd_pipeline!(RayTrace);
