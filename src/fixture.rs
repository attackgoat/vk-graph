use {
    super::{AnyResource, CommandData, Execution, Graph},
    crate::{
        cmd::SubresourceRange,
        driver::{
            self,
            accel_struct::AccelerationStructureInfo,
            buffer::{BufferInfo, BufferSubresourceRange},
            image::{ImageInfo, SampleCount},
            is_read_access,
        },
    },
    ash::vk,
    std::{
        collections::BTreeMap,
        fmt::Write as _,
        fs::write,
        io::{self, ErrorKind},
        path::Path,
    },
    vk_sync::AccessType,
};

const MAGIC: &[u8; 8] = b"VKGRFXT1";
const MAX_ITEM_COUNT: usize = 1_000_000;
const MAX_FIXTURE_BYTES: u64 = 64 * 1024 * 1024;

fn access_writes(access: AccessType) -> bool {
    !matches!(
        access,
        AccessType::Nothing
            | AccessType::CommandBufferReadNVX
            | AccessType::IndirectBuffer
            | AccessType::IndexBuffer
            | AccessType::VertexBuffer
            | AccessType::VertexShaderReadUniformBuffer
            | AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::VertexShaderReadOther
            | AccessType::MeshShaderReadUniformBuffer
            | AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::MeshShaderReadOther
            | AccessType::TaskShaderReadUniformBuffer
            | AccessType::TaskShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::TaskShaderReadOther
            | AccessType::TessellationControlShaderReadUniformBuffer
            | AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::TessellationControlShaderReadOther
            | AccessType::TessellationEvaluationShaderReadUniformBuffer
            | AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::TessellationEvaluationShaderReadOther
            | AccessType::GeometryShaderReadUniformBuffer
            | AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::GeometryShaderReadOther
            | AccessType::FragmentShaderReadUniformBuffer
            | AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::FragmentShaderReadColorInputAttachment
            | AccessType::FragmentShaderReadDepthStencilInputAttachment
            | AccessType::FragmentShaderReadOther
            | AccessType::ColorAttachmentRead
            | AccessType::DepthStencilAttachmentRead
            | AccessType::ComputeShaderReadUniformBuffer
            | AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::ComputeShaderReadOther
            | AccessType::AnyShaderReadUniformBuffer
            | AccessType::AnyShaderReadUniformBufferOrVertexBuffer
            | AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::AnyShaderReadOther
            | AccessType::TransferRead
            | AccessType::HostRead
            | AccessType::Present
            | AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer
            | AccessType::RayTracingShaderReadColorInputAttachment
            | AccessType::RayTracingShaderReadDepthStencilInputAttachment
            | AccessType::RayTracingShaderReadAccelerationStructure
            | AccessType::RayTracingShaderReadOther
            | AccessType::AccelerationStructureBuildRead
    )
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

fn invalid_input(label: &'static str, message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, format!("{label} {message}"))
}

fn meaningful_execs(command: &CommandData) -> impl Iterator<Item = &Execution> {
    command
        .execs
        .iter()
        .filter(|exec| exec.func.is_some() || exec.accesses.iter().len() != 0)
}

fn meaningful_commands(commands: &[CommandData]) -> Vec<&CommandData> {
    commands
        .iter()
        .filter(|command| meaningful_execs(command).next().is_some())
        .collect()
}

fn relationship_markdown(graph: &Graph, commands: &[&CommandData]) -> String {
    const MAX_EDGES: usize = 250;

    let mut by_resource = (0..graph.resources.len())
        .map(|_| BTreeMap::<usize, ResourceCommandAccess>::new())
        .collect::<Vec<_>>();
    let mut read_count = 0;
    let mut write_count = 0;
    let mut read_write_count = 0;

    for (command_idx, command) in commands.iter().enumerate() {
        for exec in meaningful_execs(command) {
            for (node_idx, accesses) in exec.accesses.iter() {
                for access in accesses {
                    let reads = is_read_access(access.access);
                    let writes = access_writes(access.access);
                    match (reads, writes) {
                        (true, true) => read_write_count += 1,
                        (true, false) => read_count += 1,
                        (false, true) => write_count += 1,
                        (false, false) => {}
                    }

                    let entry = by_resource[node_idx].entry(command_idx).or_default();
                    entry.access_count += 1;
                    entry.reads |= reads;
                    entry.writes |= writes;
                }
            }
        }
    }

    let mut edges = BTreeMap::<(usize, usize), OverviewEdge>::new();
    for accesses in &by_resource {
        let accesses = accesses.iter().collect::<Vec<_>>();
        for left_idx in 0..accesses.len() {
            let first_cmd = *accesses[left_idx].0;
            let first = accesses[left_idx].1;
            for second in &accesses[left_idx + 1..] {
                let second_cmd = *second.0;
                let second = second.1;
                let raw = first.writes && second.reads;
                let war = first.reads && second.writes;
                let waw = first.writes && second.writes;
                if !raw && !war && !waw {
                    continue;
                }

                let edge = edges.entry((first_cmd, second_cmd)).or_default();
                edge.access_count += first.access_count + second.access_count;
                edge.resource_count += 1;
                edge.raw |= raw;
                edge.war |= war;
                edge.waw |= waw;
            }
        }
    }

    let access_count = read_count + write_count + read_write_count;
    let displayed = edges.len().min(MAX_EDGES);
    let mut markdown = String::new();
    let _ = writeln!(markdown, "<!--");
    let _ = writeln!(markdown, "Graph fixture. Unstable dev-only format.");
    let _ = writeln!(markdown, "resources: {}", graph.resources.len());
    let _ = writeln!(markdown, "commands: {}", commands.len());
    let _ = writeln!(markdown, "accesses: {access_count}");
    let _ = writeln!(markdown, "read edges: {read_count}");
    let _ = writeln!(markdown, "write edges: {write_count}");
    let _ = writeln!(markdown, "read-write edges: {read_write_count}");
    let _ = writeln!(markdown, "compact dependencies: {}", edges.len());
    let _ = writeln!(markdown, "compact dependencies displayed: {displayed}");
    let _ = writeln!(
        markdown,
        "compact dependencies omitted: {}",
        edges.len() - displayed
    );
    let _ = writeln!(markdown, "-->\n");
    let _ = writeln!(markdown, "# Graph Relationship Overview\n");
    let _ = writeln!(markdown, "- Resources: {}", graph.resources.len());
    let _ = writeln!(markdown, "- Commands: {}", commands.len());
    let _ = writeln!(markdown, "- Accesses: {access_count}");
    let _ = writeln!(markdown, "- Compact dependencies: {}\n", edges.len());
    let _ = writeln!(markdown, "## Command Dependencies\n");
    let _ = writeln!(
        markdown,
        "Edge labels are `resources / accesses / dependency kinds`, where RAW is read-after-write, WAR is write-after-read, and WAW is write-after-write.\n"
    );
    let _ = writeln!(markdown, "```mermaid\nflowchart TD");

    let mut displayed_commands = vec![false; commands.len()];
    for (&(first, second), _) in edges.iter().take(MAX_EDGES) {
        displayed_commands[first] = true;
        displayed_commands[second] = true;
    }
    for (command_idx, command) in commands.iter().enumerate() {
        if displayed_commands[command_idx] {
            let label = command.name().replace(['"', '\n', '\r'], " ");
            let _ = writeln!(markdown, "  C{command_idx}[\"cmd {command_idx}: {label}\"]");
        }
    }
    for (&(first, second), edge) in edges.iter().take(MAX_EDGES) {
        let mut kinds = Vec::new();
        if edge.raw {
            kinds.push("RAW");
        }
        if edge.war {
            kinds.push("WAR");
        }
        if edge.waw {
            kinds.push("WAW");
        }
        let _ = writeln!(
            markdown,
            "  C{first} -- \"{} res / {} acc / {}\" --> C{second}",
            edge.resource_count,
            edge.access_count,
            kinds.join("+")
        );
    }
    let _ = writeln!(markdown, "```");

    markdown
}

fn sample_count_into_u8(sample_count: SampleCount) -> u8 {
    match sample_count {
        SampleCount::Type1 => 1,
        SampleCount::Type2 => 2,
        SampleCount::Type4 => 4,
        SampleCount::Type8 => 8,
        SampleCount::Type16 => 16,
        SampleCount::Type32 => 32,
        SampleCount::Type64 => 64,
    }
}

fn sample_count_from_u8(value: u8) -> io::Result<SampleCount> {
    match value {
        1 => Ok(SampleCount::Type1),
        2 => Ok(SampleCount::Type2),
        4 => Ok(SampleCount::Type4),
        8 => Ok(SampleCount::Type8),
        16 => Ok(SampleCount::Type16),
        32 => Ok(SampleCount::Type32),
        64 => Ok(SampleCount::Type64),
        _ => Err(invalid_data("invalid image sample count")),
    }
}

/// An unstable graph fixture used for scheduler development.
///
/// This type intentionally carries no Vulkan objects. [`Self::into_graph`] rebuilds a graph with
/// stream-argument resources and no-op command callbacks while preserving resource metadata,
/// execution boundaries, access types, and subresource ranges.
#[doc(hidden)]
#[derive(Debug)]
pub struct Fixture {
    resources: Vec<FixtureResource>,
    commands: Vec<FixtureCommand>,
}

impl Fixture {
    /// Returns the number of subresource accesses stored by this fixture.
    #[doc(hidden)]
    pub fn access_count(&self) -> usize {
        self.commands
            .iter()
            .flat_map(|cmd| &cmd.execs)
            .map(|exec| exec.accesses.len())
            .sum()
    }

    /// Returns the number of commands stored by this fixture.
    #[doc(hidden)]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Rebuilds this fixture as a graph containing no-op command callbacks.
    #[doc(hidden)]
    pub fn into_graph(self) -> Graph {
        let mut graph = Graph::new();

        for (expected_idx, resource) in self.resources.into_iter().enumerate() {
            let node_idx = match resource {
                FixtureResource::AccelerationStructure(info) => {
                    graph.bind_stream_arg_resource(AnyResource::AccelerationStructureArg(info))
                }
                FixtureResource::Buffer(info) => {
                    graph.bind_stream_arg_resource(AnyResource::BufferArg(info))
                }
                FixtureResource::Image(info) => {
                    graph.bind_stream_arg_resource(AnyResource::ImageArg(info))
                }
            };
            debug_assert_eq!(node_idx, expected_idx);
        }

        for command in self.commands {
            let mut graph_cmd = graph.begin_cmd();
            for exec in command.execs {
                for access in exec.accesses {
                    graph_cmd.push_subresource_access_index(
                        access.node_idx,
                        access.subresource,
                        access.access,
                    );
                }
                graph_cmd.record_cmd_mut(|_| {});
            }
            graph_cmd.end_cmd();
        }

        graph
    }

    /// Reads a binary graph fixture.
    #[doc(hidden)]
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len > MAX_FIXTURE_BYTES {
            return Err(invalid_data("unsupported file size"));
        }

        let bytes = std::fs::read(path)?;
        let mut reader = FixtureReader::new(&bytes);
        if reader.bytes(MAGIC.len())? != MAGIC {
            return Err(invalid_data("invalid magic"));
        }

        let resource_count = reader.count("resource count")?;
        let command_count = reader.count("command count")?;
        let mut resources = Vec::with_capacity(resource_count);
        for _ in 0..resource_count {
            resources.push(reader.read_resource()?);
        }

        let mut commands = Vec::with_capacity(command_count);
        for _ in 0..command_count {
            let exec_count = reader.count("execution count")?;
            if exec_count == 0 {
                return Err(invalid_data("empty command"));
            }

            let mut execs = Vec::with_capacity(exec_count);
            for _ in 0..exec_count {
                let access_count = reader.count("access count")?;
                let mut accesses = Vec::with_capacity(access_count);
                for _ in 0..access_count {
                    let node_idx = reader.u32()? as usize;
                    if node_idx >= resource_count {
                        return Err(invalid_data("invalid resource access"));
                    }

                    let access_value = reader.u8()?;
                    if access_value > 67 {
                        return Err(invalid_data("invalid access type"));
                    }

                    let subresource = reader.read_subresource()?;
                    resources[node_idx].validate_subresource(subresource)?;
                    accesses.push(FixtureAccess {
                        node_idx,
                        access: driver::access_type_from_u8(access_value),
                        subresource,
                    });
                }
                execs.push(FixtureExecution { accesses });
            }
            commands.push(FixtureCommand { execs });
        }

        if !reader.is_empty() {
            return Err(invalid_data("unexpected trailing bytes"));
        }

        Ok(Self {
            resources,
            commands,
        })
    }

    /// Returns the number of resources stored by this fixture.
    #[doc(hidden)]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
}

#[derive(Debug)]
struct FixtureAccess {
    node_idx: usize,
    access: AccessType,
    subresource: SubresourceRange,
}

#[derive(Debug)]
struct FixtureCommand {
    execs: Vec<FixtureExecution>,
}

#[derive(Debug)]
struct FixtureExecution {
    accesses: Vec<FixtureAccess>,
}

#[derive(Debug)]
enum FixtureResource {
    AccelerationStructure(AccelerationStructureInfo),
    Buffer(BufferInfo),
    Image(ImageInfo),
}

impl FixtureResource {
    fn validate_subresource(&self, subresource: SubresourceRange) -> io::Result<()> {
        match (self, subresource) {
            (Self::AccelerationStructure(_), SubresourceRange::AccelerationStructure) => Ok(()),
            (Self::Buffer(info), SubresourceRange::Buffer(range)) => {
                let end = if range.end == vk::WHOLE_SIZE {
                    info.size
                } else {
                    range.end
                };
                if range.start >= end || end > info.size {
                    return Err(invalid_data("invalid buffer subresource range"));
                }

                Ok(())
            }
            (Self::Image(info), SubresourceRange::Image(range)) => {
                let aspect_mask = driver::format_aspect_mask(info.format);
                if range.aspect_mask.is_empty() || !aspect_mask.contains(range.aspect_mask) {
                    return Err(invalid_data("invalid image aspect mask"));
                }

                let layers_fit = range.base_array_layer < info.array_layer_count
                    && (range.layer_count == vk::REMAINING_ARRAY_LAYERS
                        || range.layer_count > 0
                            && range
                                .base_array_layer
                                .checked_add(range.layer_count)
                                .is_some_and(|end| end <= info.array_layer_count));
                let levels_fit = range.base_mip_level < info.mip_level_count
                    && (range.level_count == vk::REMAINING_MIP_LEVELS
                        || range.level_count > 0
                            && range
                                .base_mip_level
                                .checked_add(range.level_count)
                                .is_some_and(|end| end <= info.mip_level_count));
                if !layers_fit || !levels_fit {
                    return Err(invalid_data("invalid image subresource range"));
                }

                Ok(())
            }
            _ => Err(invalid_data("subresource kind does not match resource")),
        }
    }
}

struct FixtureReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> FixtureReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn bool(&mut self) -> io::Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid_data("invalid boolean")),
        }
    }

    fn bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| invalid_data("overflow"))?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| invalid_data("underrun"))?;
        self.cursor = end;
        Ok(value)
    }

    fn count(&mut self, label: &'static str) -> io::Result<usize> {
        let count = self.u32()? as usize;
        if count > MAX_ITEM_COUNT || count > self.bytes.len().saturating_sub(self.cursor) {
            return Err(invalid_data(label));
        }
        Ok(count)
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?.try_into().expect("slice length checked"),
        ))
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("slice length checked"),
        ))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("slice length checked"),
        ))
    }

    fn read_resource(&mut self) -> io::Result<FixtureResource> {
        match self.u8()? {
            0 => Ok(FixtureResource::AccelerationStructure(
                AccelerationStructureInfo {
                    acceleration_structure_type: vk::AccelerationStructureTypeKHR::from_raw(
                        self.i32()?,
                    ),
                    size: self.u64()?,
                },
            )),
            1 => Ok(FixtureResource::Buffer(BufferInfo {
                alignment: self.u64()?,
                alloc_dedicated: self.bool()?,
                host_readable: self.bool()?,
                host_writable: self.bool()?,
                sharing_mode: vk::SharingMode::from_raw(self.i32()?),
                size: self.u64()?,
                usage: vk::BufferUsageFlags::from_raw(self.u32()?),
            })),
            2 => Ok(FixtureResource::Image(ImageInfo {
                alloc_dedicated: self.bool()?,
                array_layer_count: self.u32()?,
                depth: self.u32()?,
                flags: vk::ImageCreateFlags::from_raw(self.u32()?),
                format: vk::Format::from_raw(self.i32()?),
                height: self.u32()?,
                host_readable: self.bool()?,
                host_writable: self.bool()?,
                mip_level_count: self.u32()?,
                sample_count: sample_count_from_u8(self.u8()?)?,
                sharing_mode: vk::SharingMode::from_raw(self.i32()?),
                tiling: vk::ImageTiling::from_raw(self.i32()?),
                image_type: vk::ImageType::from_raw(self.i32()?),
                usage: vk::ImageUsageFlags::from_raw(self.u32()?),
                width: self.u32()?,
            })),
            _ => Err(invalid_data("invalid resource kind")),
        }
    }

    fn read_subresource(&mut self) -> io::Result<SubresourceRange> {
        match self.u8()? {
            0 => Ok(SubresourceRange::AccelerationStructure),
            1 => Ok(SubresourceRange::Buffer(BufferSubresourceRange {
                start: self.u64()?,
                end: self.u64()?,
            })),
            2 => Ok(SubresourceRange::Image(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::from_raw(self.u32()?),
                base_mip_level: self.u32()?,
                level_count: self.u32()?,
                base_array_layer: self.u32()?,
                layer_count: self.u32()?,
            })),
            _ => Err(invalid_data("invalid subresource kind")),
        }
    }
}

#[derive(Default)]
struct FixtureWriter(Vec<u8>);

impl FixtureWriter {
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn bytes(&mut self, value: &[u8]) {
        self.0.extend_from_slice(value);
    }

    fn count(&mut self, value: usize, label: &'static str) -> io::Result<()> {
        if value > MAX_ITEM_COUNT {
            return Err(invalid_input(label, "exceeds the fixture limit"));
        }

        let value = u32::try_from(value)
            .map_err(|_| invalid_input(label, "does not fit in the fixture format"))?;
        self.u32(value);

        Ok(())
    }

    fn finish(self) -> io::Result<Vec<u8>> {
        Self::validate_size(self.0.len())?;

        Ok(self.0)
    }

    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn validate_size(len: usize) -> io::Result<()> {
        let len = u64::try_from(len)
            .map_err(|_| invalid_input("fixture size", "does not fit in the fixture format"))?;
        if len > MAX_FIXTURE_BYTES {
            return Err(invalid_input("fixture size", "exceeds the fixture limit"));
        }

        Ok(())
    }

    fn write_resource(&mut self, resource: &AnyResource) {
        let kind = match resource {
            AnyResource::AccelerationStructure(_)
            | AnyResource::AccelerationStructureArg(_)
            | AnyResource::AccelerationStructureLease(_) => 0,
            AnyResource::Buffer(_) | AnyResource::BufferArg(_) | AnyResource::BufferLease(_) => 1,
            AnyResource::Image(_)
            | AnyResource::ImageArg(_)
            | AnyResource::ImageLease(_)
            | AnyResource::SwapchainImage(_) => 2,
        };
        self.u8(kind);
        match kind {
            0 => {
                let info = resource.expect_accel_struct_info();
                self.i32(info.acceleration_structure_type.as_raw());
                self.u64(info.size);
            }
            1 => {
                let info = resource.expect_buffer_info();
                self.u64(info.alignment);
                self.bool(info.alloc_dedicated);
                self.bool(info.host_readable);
                self.bool(info.host_writable);
                self.i32(info.sharing_mode.as_raw());
                self.u64(info.size);
                self.u32(info.usage.as_raw());
            }
            2 => {
                let info = resource.expect_image_info();
                self.bool(info.alloc_dedicated);
                self.u32(info.array_layer_count);
                self.u32(info.depth);
                self.u32(info.flags.as_raw());
                self.i32(info.format.as_raw());
                self.u32(info.height);
                self.bool(info.host_readable);
                self.bool(info.host_writable);
                self.u32(info.mip_level_count);
                self.u8(sample_count_into_u8(info.sample_count));
                self.i32(info.sharing_mode.as_raw());
                self.i32(info.tiling.as_raw());
                self.i32(info.image_type.as_raw());
                self.u32(info.usage.as_raw());
                self.u32(info.width);
            }
            _ => unreachable!(),
        }
    }

    fn write_subresource(&mut self, subresource: SubresourceRange) {
        match subresource {
            SubresourceRange::AccelerationStructure => self.u8(0),
            SubresourceRange::Buffer(range) => {
                self.u8(1);
                self.u64(range.start);
                self.u64(range.end);
            }
            SubresourceRange::Image(range) => {
                self.u8(2);
                self.u32(range.aspect_mask.as_raw());
                self.u32(range.base_mip_level);
                self.u32(range.level_count);
                self.u32(range.base_array_layer);
                self.u32(range.layer_count);
            }
        }
    }
}

impl Graph {
    /// Exports scheduler data to an unstable binary fixture and Markdown overview.
    ///
    /// The fixture contains resource descriptions, command execution boundaries, access types, and
    /// subresource ranges, but no resource contents or command callbacks. This API exists for
    /// development and benchmark fixture generation; its format may change without notice.
    #[doc(hidden)]
    pub fn export_fixture(
        &self,
        binary_path: impl AsRef<Path>,
        markdown_path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let commands = meaningful_commands(&self.cmds);
        let mut writer = FixtureWriter::default();
        writer.bytes(MAGIC);
        writer.count(self.resources.len(), "resource count")?;
        writer.count(commands.len(), "command count")?;

        for resource in self.resources.iter() {
            writer.write_resource(resource);
        }

        for command in &commands {
            let execs = meaningful_execs(command).collect::<Vec<_>>();
            writer.count(execs.len(), "execution count")?;
            for exec in execs {
                let access_count = exec
                    .accesses
                    .iter()
                    .map(|(_, accesses)| accesses.len())
                    .sum();
                writer.count(access_count, "access count")?;

                for (node_idx, accesses) in exec.accesses.iter() {
                    for access in accesses {
                        writer.u32(node_idx as u32);
                        writer.u8(driver::access_type_into_u8(access.access));
                        writer.write_subresource(access.subresource);
                    }
                }
            }
        }

        let bytes = writer.finish()?;
        write(binary_path, bytes)?;
        write(markdown_path, relationship_markdown(self, &commands))
    }

    /// Imports an unstable graph fixture as a graph with no-op callbacks.
    #[doc(hidden)]
    pub fn import_fixture(binary_path: impl AsRef<Path>) -> io::Result<Self> {
        Self::read_fixture(binary_path).map(Fixture::into_graph)
    }

    /// Reads an unstable graph fixture.
    #[doc(hidden)]
    pub fn read_fixture(binary_path: impl AsRef<Path>) -> io::Result<Fixture> {
        Fixture::read(binary_path)
    }
}

#[derive(Clone, Copy, Default)]
struct ResourceCommandAccess {
    access_count: usize,
    reads: bool,
    writes: bool,
}

#[derive(Clone, Copy, Default)]
struct OverviewEdge {
    access_count: usize,
    resource_count: usize,
    raw: bool,
    war: bool,
    waw: bool,
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::driver::{
            accel_struct::AccelerationStructureInfo, buffer::BufferInfo, image::ImageInfo,
        },
    };

    fn fixture_paths(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("vk_graph_{name}_{}_{}", std::process::id(), stamp));
        (base.with_extension("bin"), base.with_extension("md"))
    }

    fn fixture_graph() -> Graph {
        let mut graph = Graph::new();
        let buffer =
            graph.bind_stream_arg_resource(AnyResource::BufferArg(BufferInfo::device_mem(
                64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            )));
        let image =
            graph.bind_stream_arg_resource(AnyResource::ImageArg(ImageInfo::image_2d_array(
                4,
                4,
                2,
                vk::Format::R8_UINT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )));
        let accel_struct = graph.bind_stream_arg_resource(AnyResource::AccelerationStructureArg(
            AccelerationStructureInfo::blas(256),
        ));

        let mut command = graph.begin_cmd();
        command.push_subresource_access_index(
            buffer,
            SubresourceRange::Buffer(BufferSubresourceRange { start: 0, end: 32 }),
            AccessType::TransferWrite,
        );
        command.push_subresource_access_index(
            image,
            SubresourceRange::Image(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }),
            AccessType::TransferRead,
        );
        command.record_cmd_mut(|_| {});
        command.end_cmd();

        let mut command = graph.begin_cmd();
        command.push_subresource_access_index(
            accel_struct,
            SubresourceRange::AccelerationStructure,
            AccessType::RayTracingShaderReadAccelerationStructure,
        );
        command.push_subresource_access_index(
            image,
            SubresourceRange::Image(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 1,
                layer_count: 1,
            }),
            AccessType::TransferWrite,
        );
        command.record_cmd_mut(|_| {});
        command.end_cmd();

        graph
    }

    fn fixture_with_access(resource: AnyResource, subresource: SubresourceRange) -> Vec<u8> {
        let mut writer = FixtureWriter::default();
        writer.bytes(MAGIC);
        writer.count(1, "resource count").unwrap();
        writer.count(1, "command count").unwrap();
        writer.write_resource(&resource);
        writer.count(1, "execution count").unwrap();
        writer.count(1, "access count").unwrap();
        writer.u32(0);
        writer.u8(driver::access_type_into_u8(AccessType::TransferRead));
        writer.write_subresource(subresource);
        writer.finish().unwrap()
    }

    fn read_fixture_bytes(name: &str, bytes: &[u8]) -> io::Result<Fixture> {
        let (binary_path, _) = fixture_paths(name);
        std::fs::write(&binary_path, bytes).expect("unable to write test fixture");
        let result = Fixture::read(&binary_path);
        let _ = std::fs::remove_file(binary_path);
        result
    }

    #[test]
    fn fixture_round_trips() {
        let graph = fixture_graph();
        let (binary_path, markdown_path) = fixture_paths("round_trip");
        graph
            .export_fixture(&binary_path, &markdown_path)
            .expect("unable to export fixture");

        let fixture = Fixture::read(&binary_path).expect("unable to read fixture");
        assert_eq!(fixture.resource_count(), 3);
        assert_eq!(fixture.command_count(), 2);
        assert_eq!(fixture.access_count(), 4);

        let rebuilt = fixture.into_graph();
        assert_eq!(rebuilt.resources.len(), 3);
        assert_eq!(rebuilt.cmds.len(), 2);
        assert_eq!(
            rebuilt
                .cmds
                .iter()
                .flat_map(|command| &command.execs)
                .flat_map(|exec| exec.accesses.iter())
                .map(|(_, accesses)| accesses.len())
                .sum::<usize>(),
            4
        );
        assert_eq!(rebuilt.finalize().graph().cmds.len(), 2);

        let markdown = std::fs::read_to_string(&markdown_path).expect("missing Markdown fixture");
        assert!(markdown.contains("Command Dependencies"));
        assert!(markdown.contains("WAR"));

        let _ = std::fs::remove_file(binary_path);
        let _ = std::fs::remove_file(markdown_path);
    }

    #[test]
    fn real_fixtures_import_and_finalize() {
        for (name, resources, commands, accesses) in [
            ("graph-1783212230368.bin", 125, 49, 262),
            ("graph-1783212245365.bin", 114, 114, 401),
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("res/graph-fixture")
                .join(name);
            let fixture =
                Fixture::read(&path).unwrap_or_else(|err| panic!("unable to read {name}: {err}"));
            assert_eq!(fixture.resource_count(), resources);
            assert_eq!(fixture.command_count(), commands);
            assert_eq!(fixture.access_count(), accesses);
            assert_eq!(fixture.into_graph().finalize().graph().cmds.len(), commands);
        }
    }

    #[test]
    fn fixture_rejects_malformed_subresources() {
        let buffer_info = BufferInfo::device_mem(64, vk::BufferUsageFlags::TRANSFER_SRC);
        let image_info = ImageInfo::image_2d_array(
            4,
            4,
            2,
            vk::Format::R8_UINT,
            vk::ImageUsageFlags::TRANSFER_SRC,
        );
        let image_range =
            |aspect_mask, base_mip_level, level_count, base_array_layer, layer_count| {
                SubresourceRange::Image(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level,
                    level_count,
                    base_array_layer,
                    layer_count,
                })
            };

        for (name, bytes, message) in [
            (
                "mismatched_kind",
                fixture_with_access(
                    AnyResource::BufferArg(buffer_info),
                    SubresourceRange::AccelerationStructure,
                ),
                "subresource kind does not match resource",
            ),
            (
                "buffer_bounds",
                fixture_with_access(
                    AnyResource::BufferArg(buffer_info),
                    SubresourceRange::Buffer(BufferSubresourceRange { start: 32, end: 65 }),
                ),
                "invalid buffer subresource range",
            ),
            (
                "image_bounds",
                fixture_with_access(
                    AnyResource::ImageArg(image_info),
                    image_range(vk::ImageAspectFlags::COLOR, 0, 1, 1, 2),
                ),
                "invalid image subresource range",
            ),
            (
                "image_aspect",
                fixture_with_access(
                    AnyResource::ImageArg(image_info),
                    image_range(vk::ImageAspectFlags::DEPTH, 0, 1, 0, 1),
                ),
                "invalid image aspect mask",
            ),
        ] {
            let error = read_fixture_bytes(name, &bytes).expect_err("fixture should fail");
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn fixture_accepts_remaining_subresource_ranges() {
        let bytes = fixture_with_access(
            AnyResource::ImageArg(ImageInfo::image_2d_array(
                4,
                4,
                2,
                vk::Format::R8_UINT,
                vk::ImageUsageFlags::TRANSFER_SRC,
            )),
            SubresourceRange::Image(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 1,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            }),
        );

        read_fixture_bytes("remaining_ranges", &bytes).expect("fixture should be valid");
    }

    #[test]
    fn fixture_rejects_empty_commands() {
        let mut writer = FixtureWriter::default();
        writer.bytes(MAGIC);
        writer.count(0, "resource count").unwrap();
        writer.count(1, "command count").unwrap();
        writer.count(0, "execution count").unwrap();

        let bytes = writer.finish().unwrap();
        let error = read_fixture_bytes("empty_command", &bytes).expect_err("fixture should fail");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "empty command");
    }

    #[test]
    fn fixture_writer_enforces_reader_limits() {
        let mut writer = FixtureWriter::default();
        let count_error = writer
            .count(MAX_ITEM_COUNT + 1, "test count")
            .expect_err("oversized count should fail");
        assert_eq!(count_error.kind(), ErrorKind::InvalidInput);

        FixtureWriter::validate_size(MAX_FIXTURE_BYTES as usize)
            .expect("maximum size should be valid");
        let size_error = FixtureWriter::validate_size(MAX_FIXTURE_BYTES as usize + 1)
            .expect_err("oversized fixture should fail");
        assert_eq!(size_error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn relationship_fixture_rejects_invalid_input() {
        let (binary_path, _) = fixture_paths("invalid");
        std::fs::write(&binary_path, b"not a fixture").expect("unable to write invalid fixture");
        let error = Fixture::read(&binary_path).expect_err("fixture should fail");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        let _ = std::fs::remove_file(binary_path);
    }
}
