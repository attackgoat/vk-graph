//! Reusable effects and rendering utilities built on top of `vk-graph`.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod bitmap_font;
mod image_loader;
mod presenter;
mod transition;

pub use self::{
    bitmap_font::{BitmapFont, BitmapGlyphColor},
    image_loader::{ImageFormat, ImageLoader},
    presenter::{ComputePresenter, GraphicPresenter},
    transition::{Transition, TransitionPipeline},
};
