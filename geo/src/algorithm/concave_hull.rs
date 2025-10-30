use crate::bool_ops::BoolOpsNum;
use crate::convex_hull::qhull;
use crate::utils::partial_min;
use crate::{
    BoundingRect, Buffer, Coord, CoordNum, Distance, Euclidean, GeoFloat, Geometry, GeometryCollection, 
    Intersects, Length, Line, LinesIter, LineString, MultiLineString, MultiPoint, MultiPolygon, Polygon,
    coord, point,
};
use rstar::{AABB, Envelope, ParentNode, RTree, RTreeNode, RTreeNum};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, VecDeque},
};

/// Returns a polygon which covers a geometry. Unlike convex hulls, which also cover
/// their geometry, a concave hull does so while trying to further minimize its area by
/// constructing edges such that the exterior of the polygon incorporates points that would
/// be interior points in a convex hull.
///
/// This implementation is heavily based on <https://github.com/mapbox/concaveman>
/// with the addition of ensuring interior geometries are not intersected by the hull.
///
/// # Examples
/// ```
/// use geo::{line_string, MultiPoint};
/// use geo::ConcaveHull;
///
/// // a collection of points
/// let points: MultiPoint<_> = vec![
///     (0.0, 0.0), 
///     (2.0, 0.0),
///     (1.5, 1.0),
///     (2.0, 2.0),
///     (0.0, 2.0),
/// ].into();
/// 
/// let correct_hull = polygon![
///     (x: 2.0, y: 0.0),
///     (x: 1.5, y: 1.0),
///     (x: 2.0, y: 2.0),
///     (x: 0.0, y: 2.0),
///     (x: 0.0, y: 0.0),
///     (x: 2.0, y: 0.0),
/// ];
/// 
/// let hull = points.concave_hull(1.0, 0.0);
/// assert_eq!(hull, correct_hull);
/// ```
pub trait ConcaveHull {
    type Scalar: CoordNum;

    /// Create a new polygon as the concave hull of the geometry.
    /// 
    /// # Arguments
    /// * `concavity` - A relative measure of how concave the hull should be. Higher values result in a more
    ///   concave hull. Inifinity would result in a convex hull. Suggested: 2.0.
    /// 
    /// * `length_threshold` - The minimum length of constituent hull lines. Lines shorter than this will not be 
    ///   drilled down any further. Set to 0.0 if for no threshold.
    /// 
    /// # Returns
    /// * A `Polygon` representing the concave hull of the geometry.
    fn concave_hull(&self, concavity: Self::Scalar, length_threshold: Self::Scalar) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: Self::Scalar, length_threshold: Self::Scalar, dig_rule: DigRule) -> Polygon<Self::Scalar>;
}

pub enum DigRule {
    NearestHullLine,
    NearestDiggableLine,
}

impl<T> ConcaveHull for MultiPoint<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.iter().map(|point| point.0).collect();
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, DigRule::NearestHullLine, None, None)
    }
}

impl<T> ConcaveHull for Polygon<T>
where
    T: BoolOpsNum + GeoFloat + RTreeNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        add_polygon(self, &mut coords, &mut lines, &mut buffered_polygons);
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, dig_rule, Some(lines), Some(buffered_polygons))
    }
}

impl<T> ConcaveHull for MultiPolygon<T>
where
    T: BoolOpsNum + GeoFloat + RTreeNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        for polygon in self.0.iter() {
            add_polygon(polygon, &mut coords, &mut lines, &mut buffered_polygons);
        }
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, dig_rule, Some(lines), Some(buffered_polygons))
    }
}

impl<T> ConcaveHull for LineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let lines: Vec<Line<T>> = self.lines().collect();
        concave_hull_with_dig_rule(&mut self.0.clone(), concavity, length_threshold, dig_rule, Some(lines), None)
    }
}

impl<T> ConcaveHull for MultiLineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }
    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.iter().flat_map(|elem| elem.0.clone()).collect();
        let lines: Vec<Line<T>> = self.iter().flat_map(|elem| elem.lines()).collect();
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, dig_rule, Some(lines), None)
    }
}

impl<T> ConcaveHull for GeometryCollection<T>
where
    T: BoolOpsNum + GeoFloat + RTreeNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }
    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        add_geometry_collection(self, &mut coords, &mut lines, &mut buffered_polygons);
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, dig_rule, Some(lines), Some(buffered_polygons))
    }
}

impl<T> ConcaveHull for Vec<Coord<T>>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }

    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.clone();
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, DigRule::NearestHullLine, None, None)
    }
}

impl<T> ConcaveHull for [Coord<T>]
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        self.concave_hull_with_dig_rule(concavity, length_threshold, DigRule::NearestHullLine)
    }
    fn concave_hull_with_dig_rule(&self, concavity: T, length_threshold: T, dig_rule: DigRule) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.to_vec();
        concave_hull_with_dig_rule(&mut coords, concavity, length_threshold, DigRule::NearestHullLine, None, None)
    }
}

fn add_geometry_collection<T>(
    collection: &GeometryCollection<T>,
    coords: &mut Vec<Coord<T>>,
    lines: &mut Vec<Line<T>>,
    buffered_polygons: &mut Vec<Polygon<T>>,
)
where
    T: BoolOpsNum + GeoFloat + RTreeNum + 'static,
{
    for geometry in collection.0.iter() {
        match geometry {
            Geometry::Point(point) => {
                coords.push(point.0);
            }
            Geometry::MultiPoint(multi_point) => {
                coords.extend(multi_point.iter().map(|p| p.0));
            }
            Geometry::Line(line) => {
                coords.push(line.start);
                coords.push(line.end);
                lines.push(*line);
            }
            Geometry::LineString(line_string) => {
                coords.extend(line_string.0.iter());
                lines.extend(line_string.lines());
            }
            Geometry::MultiLineString(multi_line_string) => {
                for line_string in multi_line_string.iter() {
                    coords.extend(line_string.0.iter());
                    lines.extend(line_string.lines());
                }
            }
            Geometry::Polygon(polygon) => {
                add_polygon(polygon, coords, lines, buffered_polygons);
            }
            Geometry::MultiPolygon(multi_polygon) => {
                for polygon in multi_polygon.iter() {
                    add_polygon(polygon, coords, lines, buffered_polygons);
                }
            }
            Geometry::Triangle(triangle) => {
                add_polygon(&Polygon::from(*triangle), coords, lines, buffered_polygons);
            }
            Geometry::Rect(rect) => {
                add_polygon(&Polygon::from(*rect), coords, lines, buffered_polygons);
            }
            Geometry::GeometryCollection(nested_collection) => {
                add_geometry_collection(nested_collection, coords, lines, buffered_polygons);
            }
        }
    }
}

fn add_polygon<T>(
    polygon: &Polygon<T>,
    coords: &mut Vec<Coord<T>>,
    lines: &mut Vec<Line<T>>,
    buffered_polygons: &mut Vec<Polygon<T>>,
) 
where 
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
{
    coords.extend(polygon.exterior().0.iter().skip(1));
    lines.extend(polygon.exterior().lines_iter());

    // Buffer the interior polygons by a small fraction of the polygon size
    let bounding_rect = polygon.bounding_rect().unwrap();
    let buffer_distance = partial_min(bounding_rect.width(), bounding_rect.height()) * T::from(-1e-6).unwrap();
    buffered_polygons.extend(polygon.buffer(buffer_distance).0);
}

enum RTreeNodeRef<'a, T> 
where 
    T: GeoFloat + RTreeNum,
{
    Parent(&'a ParentNode<Coord<T>>),
    Leaf(&'a Coord<T>),
}

struct QueueItem<'a, T> 
where 
    T: GeoFloat + RTreeNum,
{
    tree_node: RTreeNodeRef<'a, T>,
    distance: T,
}

impl<'a, T> Ord for QueueItem<'a, T> 
where 
    T: GeoFloat + RTreeNum,
{
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}
impl<'a, T> PartialOrd for QueueItem<'a, T> 
where 
    T: GeoFloat + RTreeNum,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a, T> PartialEq for QueueItem<'a, T> 
where 
    T: GeoFloat + RTreeNum,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl<'a, T> Eq for QueueItem<'a, T> where T: GeoFloat + RTreeNum {}

fn is_interior_line<T>(
    line: &Line<T>,
    interior_lines_tree: &Option<RTree<Line<T>>>,
) -> bool
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = interior_lines_tree {
        tree.contains(line) || tree.contains(&Line::new(line.end, line.start))
    } else {
        false
    }
}

fn line_to_bbox_distance<T>(line: &Line<T>, aabb: &AABB<Coord<T>>) -> T
where
    T: GeoFloat + RTreeNum,
{
    if aabb.contains_point(&line.start) || aabb.contains_point(&line.end) {
        return T::zero();
    }
    let c1 = coord!{x: aabb.lower().x, y: aabb.lower().y};
    let c2 = coord!{x: aabb.lower().x, y: aabb.upper().y};
    let c3 = coord!{x: aabb.upper().x, y: aabb.upper().y};
    let c4 = coord!{x: aabb.upper().x, y: aabb.lower().y};
    let d1 = Euclidean.distance(line, &Line::new(c1, c4));
    if d1 == T::zero() {
        return d1;
    }
    let d2 = Euclidean.distance(line, &Line::new(c1, c2));
    if d2 == T::zero() {
        return d2;
    }
    let d3 = Euclidean.distance(line, &Line::new(c4, c3));
    if d3 == T::zero() {
        return d3;
    }
    let d4 = Euclidean.distance(line, &Line::new(c2, c3));
    if d4 == T::zero() {
        return d4;
    }
    partial_min(partial_min(d1, d2), partial_min(d3, d4))
}

fn no_hull_intersections<T>(line: &Line<T>, current_hull_tree: &RTree<Line<T>>) -> bool
where 
    T: GeoFloat + RTreeNum,
{
    let min_x = T::min(line.start.x, line.end.x);
    let max_x = T::max(line.start.x, line.end.x);
    let min_y = T::min(line.start.y, line.end.y);
    let max_y = T::max(line.start.y, line.end.y);
    let bbox = AABB::from_corners(point!([min_x, min_y]), point!([max_x, max_y]));
    let hull_lines = current_hull_tree.locate_in_envelope_intersecting(&bbox);
    for hull_line in hull_lines {
        if hull_line.start == line.start || hull_line.start == line.end ||
           hull_line.end == line.start || hull_line.end == line.end {
            continue;
        }
        if line.intersects(hull_line) {
            return false;
        }
    }
    true
}

fn shifted_intersects<T>(line: &Line<T>, interior_line: &Line<T>) -> bool 
where
    T: GeoFloat,
{
    // Shift the line each way along the x and y axis to check if the line is only 
    // touching the interior line.
    let shift = Euclidean.length(line) * T::from(1e-6).unwrap();

    interior_line.intersects(&Line::new(
        point!(x: line.start.x + shift, y: line.start.y),
        point!(x: line.end.x + shift, y: line.end.y),
    )) && 
    interior_line.intersects(&Line::new(
        point!(x: line.start.x - shift, y: line.start.y),
        point!(x: line.end.x - shift, y: line.end.y),
    )) && 
    interior_line.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y + shift),
        point!(x: line.end.x, y: line.end.y + shift),
    )) && 
    interior_line.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y - shift),
        point!(x: line.end.x, y: line.end.y - shift),
    ))
}

fn no_interior_line_intersections<T>(line: &Line<T>, tree: &Option<RTree<Line<T>>>) -> bool 
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = tree {
        let min_x = T::min(line.start.x, line.end.x);
        let max_x = T::max(line.start.x, line.end.x);
        let min_y = T::min(line.start.y, line.end.y);
        let max_y = T::max(line.start.y, line.end.y);
        let bbox = AABB::from_corners(point!([min_x, min_y]), point!([max_x, max_y]));
        let lines = tree.locate_in_envelope_intersecting(&bbox);
        for interior_line in lines {
            if shifted_intersects(line, interior_line) {
                return false;
            }
        }
    }
    true
}

fn no_interior_polygon_intersections<T>(line: &Line<T>, tree: &Option<RTree<Polygon<T>>>) -> bool 
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = tree {
        let min_x = T::min(line.start.x, line.end.x);
        let max_x = T::max(line.start.x, line.end.x);
        let min_y = T::min(line.start.y, line.end.y);
        let max_y = T::max(line.start.y, line.end.y);
        let bbox = AABB::from_corners(point!([min_x, min_y]), point!([max_x, max_y]));
        let interior_polygons = tree.locate_in_envelope_intersecting(&bbox);
        for interior_polygon in interior_polygons {
            if interior_polygon.intersects(line) {
                return false;
            }
        }
    }
    true
}

fn find_candidate<T>(
    line: &Line<T>,
    max_length: &T,
    interior_points_tree: &RTree<Coord<T>>,
    current_hull_tree: &RTree<Line<T>>,
    interior_lines_tree: &Option<RTree<Line<T>>>,
    buffered_polygon_tree: &Option<RTree<Polygon<T>>>,
) -> Option<Coord<T>>
where
    T: GeoFloat + RTreeNum,
{
    // Initialize priority queue with R-tree root node
    let mut queue: BinaryHeap<QueueItem<T>> = BinaryHeap::new();
    queue.push(QueueItem {
        tree_node: RTreeNodeRef::Parent(interior_points_tree.root()),
        distance: T::zero(),
    });

    // Perform depth first search through R-tree
    while let Some(node) = queue.pop() {
        match node.tree_node {
            RTreeNodeRef::Parent(parent) => {
                for child in parent.children() {
                    match child {
                        RTreeNode::Parent(p) => {
                            let envelope = p.envelope();
                            let distance = line_to_bbox_distance(line, &envelope);
                            if distance <= *max_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodeRef::Parent(p),
                                    distance,
                                });
                            }
                        }
                        RTreeNode::Leaf(l) => {
                            let distance = Euclidean.distance(*l, line);
                            if distance <= *max_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodeRef::Leaf(l),
                                    distance,
                                });
                            }
                        }
                    }
                }
            }
            RTreeNodeRef::Leaf(leaf) => {
                // Skip all points that are closer to another hull line
                // Note: This is the key divergence from the mapbox/concaveman algorithm which 
                // checks this for the hull lines either side of the current line
                if let Some(nearest_line) = current_hull_tree.nearest_neighbor(&point![x: leaf.x, y: leaf.y]) {
                    if nearest_line == line || nearest_line == &Line::new(line.end, line.start) {
                        let start_line = Line::new(line.start, *leaf);
                        let end_line = Line::new(*leaf, line.end);

                        // Skip candidate if it would cause intersections with hull lines or interior geometries
                        if no_hull_intersections(&start_line, current_hull_tree) 
                            && no_hull_intersections(&end_line, current_hull_tree)
                            && no_interior_line_intersections(&start_line, interior_lines_tree) 
                            && no_interior_line_intersections(&end_line, interior_lines_tree)
                            && no_interior_polygon_intersections(&start_line, buffered_polygon_tree) 
                            && no_interior_polygon_intersections(&end_line, buffered_polygon_tree)
                        {
                            return Some(*leaf);
                        }
                    }
                }
            }
        }
    }
    None
}

fn order_concave_hull<T>(concave_lines: HashMap<usize, (Line<T>, usize)>, break_i: usize) -> LineString<T>
where
    T: GeoFloat,
{
    let mut ordered_coords: Vec<Coord<T>> = vec![];
    let mut current_i = 0;
    ordered_coords.push(concave_lines.get(&current_i).unwrap().0.start);

    while current_i != break_i {
        if let Some((line, next_i)) = concave_lines.get(&current_i) {
            ordered_coords.push(line.end);
            current_i = *next_i;
        } else {
            break;
        }
    }
    LineString::from(ordered_coords)
}

fn remove_interior_point<T>(coord: &Coord<T>, tree: &mut RTree<Coord<T>>)
where
    T: GeoFloat + RTreeNum,
{
    // TODO: Handle removing potential duplicate points better
    // Currently decreases performance by ~5%
    let n = tree.nearest_neighbors(coord).len();
    for _ in 0..n {
        tree.remove(coord);
    }
}

struct LineQueueItem<T: GeoFloat + RTreeNum> {
    line: Line<T>,
    i: usize,
    next_i: usize,
}

fn concave_hull_with_dig_rule<T>(
    coords: &mut [Coord<T>],
    concavity: T,
    length_threshold: T, 
    dig_rule: DigRule,
    interior_lines: Option<Vec<Line<T>>>,
    buffered_polygons: Option<Vec<Polygon<T>>>,
) -> Polygon<T>
where
    T: GeoFloat + RTreeNum,
{
    // Ensure concavity is non-negative
    let concavity: T = T::max(T::zero(), concavity);

    // Compute initial convex hull
    let hull = qhull::quick_hull(coords);
    
    // Returns convex hull if less than 4 points
    if coords.len() < 4 {
        return Polygon::new(hull, vec![]);
    }

    // Build R-trees for interior points, hull lines and interior geometries
    let mut interior_points_tree: RTree<Coord<T>> = RTree::bulk_load(coords.to_owned());
    let mut current_hull_tree: RTree<Line<T>> = RTree::bulk_load(hull.lines().collect());
    let interior_lines_tree: Option<RTree<Line<T>>> = interior_lines.map(RTree::bulk_load);
    let buffered_polygon_tree: Option<RTree<Polygon<T>>> = buffered_polygons.map(RTree::bulk_load);

    let mut concave_lines: HashMap<usize, (Line<T>, usize)> = HashMap::new();
    let mut line_queue: VecDeque<LineQueueItem<T>> = VecDeque::new();

    // Populate line queue with initial hull lines
    for (i, line) in hull.lines().enumerate() {
        if !is_interior_line(&line, &interior_lines_tree) {
            line_queue.push_back(LineQueueItem {
                line,
                i,
                next_i: i + 1,
            });
        } else {
            concave_lines.insert(i, (line, i + 1));
            if matches!(dig_rule, DigRule::NearestDiggableLine) {
                current_hull_tree.remove(&line);
            }
        }

        // Remove hull points from interior points
        if i == 0 {
            remove_interior_point(&line.start, &mut interior_points_tree);
        }
        remove_interior_point(&line.end, &mut interior_points_tree);
    }

    // Set current hull line index for new lines
    let mut current_i = hull.0.len() + 1;

    while let Some(line_queue_item) = line_queue.pop_front() {
        let line = line_queue_item.line;
        let length = Euclidean.length(&line);
        
        // Only consider drilling down if line length exceeds threshold
        if length > length_threshold {
            let max_length = length / concavity;
            
            if let Some(candidate_point) = find_candidate(
                &line,
                &max_length,
                &interior_points_tree,
                &current_hull_tree,
                &interior_lines_tree,
                &buffered_polygon_tree,
            ) {
                let start_line = Line::new(line.start, candidate_point);
                let end_line = Line::new(candidate_point, line.end);
                
                if partial_min(Euclidean.length(&start_line), Euclidean.length(&end_line)) < max_length {
                    // Remove candidate point from interior points
                    remove_interior_point(&candidate_point, &mut interior_points_tree);
                    
                    // Update current hull tree
                    current_hull_tree.remove(&line);
                    current_hull_tree.insert(start_line);
                    current_hull_tree.insert(end_line);
                    
                    // Add new lines either directly to concave lines or to queue
                    if !is_interior_line(&start_line, &interior_lines_tree) {
                        line_queue.push_back(LineQueueItem {
                            line: start_line,
                            i: line_queue_item.i, // inherit old line i
                            next_i: current_i, // next_i becomes end line's i
                        });
                    } else {
                        concave_lines.insert(line_queue_item.i, (start_line, current_i));
                        if matches!(dig_rule, DigRule::NearestDiggableLine) {
                            current_hull_tree.remove(&start_line);
                        }
                    }

                    if !is_interior_line(&end_line, &interior_lines_tree) {
                        line_queue.push_back(LineQueueItem {
                            line: end_line,
                            i: current_i,
                            next_i: line_queue_item.next_i, // inherit old line's next_i
                        });
                    } else {
                        concave_lines.insert(current_i, (end_line, line_queue_item.next_i));
                        if matches!(dig_rule, DigRule::NearestDiggableLine) {
                            current_hull_tree.remove(&end_line);
                        }
                    }
                    
                    // Increment current_i for next potential new line
                    current_i += 1;
                    continue;
                }
            }
        }
        
        concave_lines.insert(line_queue_item.i, (line, line_queue_item.next_i));
    }
    Polygon::new(order_concave_hull(concave_lines, hull.0.len()), vec![])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{coord, polygon};

    #[test]
    fn test_empty_coords() {
        let coords: Vec<Coord<f64>> = vec![];
        let hull = coords.concave_hull(2.0, 0.0);
        assert!(hull.exterior().0.is_empty());
    }

    #[test]
    fn test_concavity() {
        let polygon: Polygon<f64> = polygon![
            (x: 10.0, y: 9.0),
            (x: 10.0, y: 10.0),
            (x: 0.0, y: 10.0),
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 9.0),
            (x: 10.0, y: 9.0),
        ];
        let hull_1 = polygon.concave_hull(1.0, 0.0);
        assert_eq!(hull_1.exterior().0.len(), 7);
        
        let hull_2 = polygon.concave_hull(2.0, 0.0);
        assert_eq!(hull_2.exterior().0.len(), 6);
    }

    #[test]
    fn test_length_threshold() {
        let coords: Vec<Coord<f64>> = vec![
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 2.0, y: 0.0 },
            coord! { x: 1.5, y: 1.0 },
            coord! { x: 2.0, y: 2.0 },
            coord! { x: 0.0, y: 2.0 },
        ];

        // The correct ordering is dependent on qhull's ccw output
        let correct_with_threshold = polygon![
            (x: 2.0, y: 0.0),
            (x: 2.0, y: 2.0),
            (x: 0.0, y: 2.0),
            (x: 0.0, y: 0.0),
            (x: 2.0, y: 0.0),
        ];
        let hull_with_threshold = coords.concave_hull(1.0, 3.0);
        assert_eq!(hull_with_threshold, correct_with_threshold);

        let correct_without_threshold = polygon![
            (x: 2.0, y: 0.0),
            (x: 1.5, y: 1.0),
            (x: 2.0, y: 2.0),
            (x: 0.0, y: 2.0),
            (x: 0.0, y: 0.0),
            (x: 2.0, y: 0.0),
        ];
        let hull_without_threshold = coords.concave_hull(1.0, 0.0);
        assert_eq!(hull_without_threshold, correct_without_threshold);
    }

    #[test]
    fn test_multipolygon() {
        let multipolygon: MultiPolygon<f64> = vec![
            polygon![
                (x: 0.0, y: 0.0),
                (x: 0.0, y: 2.0),
                (x: 2.0, y: 2.0),
                (x: 2.0, y: 0.0),
                (x: 0.0, y: 0.0),
            ],
            polygon![
                (x: 3.0, y: 0.0),
                (x: 3.0, y: 5.0),
                (x: 4.0, y: 5.0),
                (x: 4.0, y: 0.0),
                (x: 3.0, y: 0.0),
            ],
        ].into();
        let hull = multipolygon.concave_hull(2.0, 0.0);
        let correct_hull = polygon![
            (x: 4.0, y: 0.0),
            (x: 4.0, y: 5.0),
            (x: 3.0, y: 5.0),
            (x: 2.0, y: 2.0),
            (x: 0.0, y: 2.0),
            (x: 0.0, y: 0.0),
            (x: 2.0, y: 0.0),
            (x: 3.0, y: 0.0),
            (x: 4.0, y: 0.0),
        ];
        assert_eq!(hull, correct_hull);

    }

    // #[test]
    // fn test_single_point() {
    //     let coords: Vec<Coord<f64>> = vec![coord! { x: 1.0, y: 1.0 }];
    //     let hull = coords.concave_hull(None, None);
    //     assert_eq!(hull, Polygon::new(LineString::from()));
    // }

    // #[test]
    // fn test_straight_line() {
    //     let linestring: LineString<f64> = line_string![
    //         (x: 0.0, y: 0.0),
    //         (x: 2.0, y: 2.0),
    //         (x: 6.0, y: 6.0),
    //     ];
    //     let hull = linestring.concave_hull(None, None);
    // }

    // #[test]
    // fn test_square() {
    //     let square: Polygon<f64> = polygon![
    //         (x: 0.0, y: 0.0),
    //         (x: 0.0, y: 2.0),
    //         (x: 2.0, y: 2.0),
    //         (x: 2.0, y: 0.0),
    //         (x: 0.0, y: 0.0),
    //     ];
    //     let hull = square.concave_hull(None, None);
    //     assert_eq!(hull, square);
    // }
}