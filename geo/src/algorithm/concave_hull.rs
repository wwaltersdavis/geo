use crate::algorithm::Kernel;
use crate::coordinate_position::CoordPos;
use crate::convex_hull::qhull;
use crate::utils::partial_min;
use crate::{
    Coord, CoordinatePosition, CoordNum, Distance, Euclidean, GeoFloat, Geometry, GeometryCollection, Intersects,
    Length, Line, LineString, LinesIter, MultiLineString, MultiPoint, MultiPolygon, Orientation,
    Polygon, coord, point,
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
/// This implementation is a version of <https://github.com/mapbox/concaveman>
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
    ///   concave hull. Inifinity would result in a convex hull. 2.0 results in a relatively detailed shape.
    ///
    /// * `length_threshold` - The minimum length of constituent hull lines. Lines shorter than this will not be
    ///   drilled down any further. Set to 0.0 if for no threshold.
    ///
    /// # Returns
    /// * A `Polygon` representing the concave hull of the geometry.
    fn concave_hull(
        &self,
        concavity: Self::Scalar,
        length_threshold: Self::Scalar,
    ) -> Polygon<Self::Scalar>;
}

impl<T> ConcaveHull for MultiPoint<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<T> {
        let mut coords: Vec<Coord<T>> = self.iter().map(|point| point.0).collect();
        concave_hull(&mut coords, concavity, length_threshold, None, None)
    }
}

impl<T> ConcaveHull for Polygon<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut polygons: Vec<Polygon<T>> = Vec::new();
        add_polygon(self, &mut coords, &mut lines, &mut polygons);
        concave_hull(
            &mut coords,
            concavity,
            length_threshold,
            Some(lines),
            Some(polygons),
        )
    }
}

impl<T> ConcaveHull for MultiPolygon<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut polygons: Vec<Polygon<T>> = Vec::new();
        for polygon in self.0.iter() {
            add_polygon(polygon, &mut coords, &mut lines, &mut polygons);
        }
        concave_hull(
            &mut coords,
            concavity,
            length_threshold,
            Some(lines),
            Some(polygons),
        )
    }
}

impl<T> ConcaveHull for LineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let lines: Vec<Line<T>> = self.lines().collect();
        concave_hull(
            &mut self.0.clone(),
            concavity,
            length_threshold,
            Some(lines),
            None,
        )
    }
}

impl<T> ConcaveHull for MultiLineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<T> {
        let mut coords: Vec<Coord<T>> = self.iter().flat_map(|elem| elem.0.clone()).collect();
        let lines: Vec<Line<T>> = self.iter().flat_map(|elem| elem.lines()).collect();
        concave_hull(&mut coords, concavity, length_threshold, Some(lines), None)
    }
}

impl<T> ConcaveHull for GeometryCollection<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut lines: Vec<Line<T>> = Vec::new();
        let mut polygons: Vec<Polygon<T>> = Vec::new();
        add_geometry_collection(self, &mut coords, &mut lines, &mut polygons);
        concave_hull(
            &mut coords,
            concavity,
            length_threshold,
            Some(lines),
            Some(polygons),
        )
    }
}

impl<T> ConcaveHull for Vec<Coord<T>>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.clone();
        concave_hull(&mut coords, concavity, length_threshold, None, None)
    }
}

impl<T> ConcaveHull for [Coord<T>]
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: T, length_threshold: T) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = self.to_vec();
        concave_hull(&mut coords, concavity, length_threshold, None, None)
    }
}

fn add_polygon<T>(
    polygon: &Polygon<T>,
    coords: &mut Vec<Coord<T>>,
    lines: &mut Vec<Line<T>>,
    polygons: &mut Vec<Polygon<T>>,
) where
    T: GeoFloat + RTreeNum,
{
    coords.extend(polygon.exterior().0.iter().skip(1));
    lines.extend(polygon.exterior().lines_iter());
    polygons.push(Polygon::new(polygon.exterior().clone(), vec![]));
}

fn add_geometry_collection<T>(
    collection: &GeometryCollection<T>,
    coords: &mut Vec<Coord<T>>,
    lines: &mut Vec<Line<T>>,
    polygons: &mut Vec<Polygon<T>>,
) where
    T: GeoFloat + RTreeNum,
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
                add_polygon(polygon, coords, lines, polygons);
            }
            Geometry::MultiPolygon(multi_polygon) => {
                for polygon in multi_polygon.iter() {
                    add_polygon(polygon, coords, lines, polygons);
                }
            }
            Geometry::Triangle(triangle) => {
                add_polygon(&Polygon::from(*triangle), coords, lines, polygons);
            }
            Geometry::Rect(rect) => {
                add_polygon(&Polygon::from(*rect), coords, lines, polygons);
            }
            Geometry::GeometryCollection(nested_collection) => {
                add_geometry_collection(nested_collection, coords, lines, polygons);
            }
        }
    }
}

enum RTreeNodeRef<'a, T>
where
    T: GeoFloat + RTreeNum,
{
    Parent(&'a ParentNode<Coord<T>>),
    Leaf(&'a Coord<T>),
}

struct NodeQueueItem<'a, T>
where
    T: GeoFloat + RTreeNum,
{
    tree_node: RTreeNodeRef<'a, T>,
    distance: T,
}

impl<'a, T> Ord for NodeQueueItem<'a, T>
where
    T: GeoFloat + RTreeNum,
{
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}
impl<'a, T> PartialOrd for NodeQueueItem<'a, T>
where
    T: GeoFloat + RTreeNum,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a, T> PartialEq for NodeQueueItem<'a, T>
where
    T: GeoFloat + RTreeNum,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl<'a, T> Eq for NodeQueueItem<'a, T> where T: GeoFloat + RTreeNum {}

fn is_interior_line<T>(line: &Line<T>, interior_lines_tree: &Option<RTree<Line<T>>>) -> bool
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
    let c1 = coord! {x: aabb.lower().x, y: aabb.lower().y};
    let c2 = coord! {x: aabb.lower().x, y: aabb.upper().y};
    let c3 = coord! {x: aabb.upper().x, y: aabb.upper().y};
    let c4 = coord! {x: aabb.upper().x, y: aabb.lower().y};
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
        if hull_line.start == line.start
            || hull_line.start == line.end
            || hull_line.end == line.start
            || hull_line.end == line.end
        {
            continue;
        }
        if line.intersects(hull_line) {
            return false;
        }
    }
    true
}

fn transverse_intersection<T>(line_a: &Line<T>, line_b: &Line<T>) -> bool
where
    T: GeoFloat,
{
    // Only return true if they two lines have a transverse intersection.
    // Otherwise if they do not intersect or intersect tangentially return false.

    // Get orientation of line_a's start/end with respect to line_b.
    let check_1_1 = T::Ker::orient2d(line_b.start, line_b.end, line_a.start);
    let check_1_2 = T::Ker::orient2d(line_b.start, line_b.end, line_a.end);

    // If both orientations are the same or either orientation is collinear, this is not breaking the line.
    if check_1_1 == check_1_2
        || check_1_1 == Orientation::Collinear
        || check_1_2 == Orientation::Collinear
    {
        return false;
    }

    // Get orientation of line_b's start/end with respect to line_a.
    let check_2_1 = T::Ker::orient2d(line_a.start, line_a.end, line_b.start);
    let check_2_2 = T::Ker::orient2d(line_a.start, line_a.end, line_b.end);

    // Return true only if the orientations are different and neither is collinear.
    check_2_1 != check_2_2
        && check_2_1 != Orientation::Collinear
        && check_2_2 != Orientation::Collinear
}

fn breaks_interior_lines<T>(line: &Line<T>, tree: &Option<RTree<Line<T>>>) -> bool
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
            if interior_line.start == line.start
                || interior_line.start == line.end
                || interior_line.end == line.start
                || interior_line.end == line.end
            {
                continue;
            }
            if transverse_intersection(line, interior_line) {
                return true;
            }
        }
    }
    false
}

fn breaks_interior_polygons<T>(line: &Line<T>, tree: &Option<RTree<Polygon<T>>>) -> bool
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
            // Only need to check if either end of the line is inside the polygon.
            // The intersections with interior lines is covered with `breaks_interior_lines`.
            if interior_polygon.coordinate_position(&line.start) == CoordPos::Inside
                || interior_polygon.coordinate_position(&line.end) == CoordPos::Inside
            {
                return true;
            }
        }
    }
    false
}

fn find_candidate<T>(
    line_queue_item: &LineQueueItem<T>,
    max_length: &T,
    hull_lines: &HashMap<usize, Line<T>>,
    interior_points_tree: &RTree<Coord<T>>,
    current_hull_tree: &RTree<Line<T>>,
    interior_lines_tree: &Option<RTree<Line<T>>>,
    interior_polygon_tree: &Option<RTree<Polygon<T>>>,
) -> Option<Coord<T>>
where
    T: GeoFloat + RTreeNum,
{
    let line = hull_lines.get(&line_queue_item.i).unwrap();

    // Initialize priority queue with R-tree root node
    let mut queue: BinaryHeap<NodeQueueItem<T>> = BinaryHeap::new();
    queue.push(NodeQueueItem {
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
                                queue.push(NodeQueueItem {
                                    tree_node: RTreeNodeRef::Parent(p),
                                    distance,
                                });
                            }
                        }
                        RTreeNode::Leaf(l) => {
                            let distance = Euclidean.distance(*l, line);
                            if distance <= *max_length {
                                queue.push(NodeQueueItem {
                                    tree_node: RTreeNodeRef::Leaf(l),
                                    distance,
                                });
                            }
                        }
                    }
                }
            }
            RTreeNodeRef::Leaf(leaf) => {
                // Skip candidate points that as close to adjacent edges
                if node.distance
                    >= Euclidean.distance(*leaf, hull_lines.get(&line_queue_item.prev_i).unwrap())
                    || node.distance
                        >= Euclidean
                            .distance(*leaf, hull_lines.get(&line_queue_item.next_i).unwrap())
                {
                    continue;
                }
                let start_line = Line::new(line.start, *leaf);
                let end_line = Line::new(*leaf, line.end);

                // Skip candidate point if it would cause intersections with hull lines or interior geometries
                if no_hull_intersections(&start_line, current_hull_tree)
                    && no_hull_intersections(&end_line, current_hull_tree)
                    && !breaks_interior_lines(&start_line, interior_lines_tree)
                    && !breaks_interior_lines(&end_line, interior_lines_tree)
                    && !breaks_interior_polygons(&start_line, interior_polygon_tree)
                    && !breaks_interior_polygons(&end_line, interior_polygon_tree)
                {
                    return Some(*leaf);
                }
            }
        }
    }
    None
}

fn order_concave_hull<T>(
    hull_order: HashMap<usize, usize>,
    hull_lines: HashMap<usize, Line<T>>,
) -> LineString<T>
where
    T: GeoFloat,
{
    let mut ordered_coords: Vec<Coord<T>> = vec![];
    let mut current_i = 0;
    ordered_coords.push(hull_lines.get(&current_i).unwrap().start);

    for _ in 0..hull_order.len() {
        let next_i = hull_order.get(&current_i).unwrap();
        let line = hull_lines.get(&current_i).unwrap();
        ordered_coords.push(line.end);
        current_i = *next_i;
    }
    LineString::from(ordered_coords)
}

fn remove_interior_point<T>(coord: &Coord<T>, tree: &mut RTree<Coord<T>>)
where
    T: GeoFloat + RTreeNum,
{
    let n = tree.nearest_neighbors(coord).len();
    for _ in 0..n {
        tree.remove(coord);
    }
}

struct LineQueueItem<T: GeoFloat + RTreeNum> {
    line: Line<T>,
    i: usize,
    prev_i: usize,
    next_i: usize,
}

fn concave_hull<T>(
    coords: &mut [Coord<T>],
    concavity: T,
    length_threshold: T,
    interior_lines: Option<Vec<Line<T>>>,
    interior_polygons: Option<Vec<Polygon<T>>>,
) -> Polygon<T>
where
    T: GeoFloat + RTreeNum,
{
    // Ensure concavity is non-negative
    let concavity: T = T::max(T::zero(), concavity);

    // Compute initial convex hull
    let hull = qhull::quick_hull(coords);

    // Return convex hull if less than 4 points
    if coords.len() < 4 {
        return Polygon::new(hull, vec![]);
    }

    // Build R-trees for interior points and hull lines
    let mut interior_points_tree: RTree<Coord<T>> = RTree::bulk_load(coords.to_owned());
    let mut current_hull_tree: RTree<Line<T>> = RTree::bulk_load(hull.lines().collect());

    // Build R-trees for interior lines and polygons if provided
    let interior_lines_tree: Option<RTree<Line<T>>> = interior_lines.map(RTree::bulk_load);
    let interior_polygon_tree: Option<RTree<Polygon<T>>> = interior_polygons.map(RTree::bulk_load);

    let mut line_queue: VecDeque<LineQueueItem<T>> = VecDeque::new();
    let mut hull_lines: HashMap<usize, Line<T>> = HashMap::new();
    let mut hull_order: HashMap<usize, usize> = HashMap::new();

    // Populate line queue with initial hull lines
    for (i, line) in hull.lines().enumerate() {
        hull_lines.insert(i, line);
        let next_i = if i == hull.0.len() - 2 { 0 } else { i + 1 };
        if !is_interior_line(&line, &interior_lines_tree) {
            line_queue.push_back(LineQueueItem {
                line,
                i,
                prev_i: if i == 0 { hull.0.len() - 2 } else { i - 1 },
                next_i,
            });
        } else {
            hull_order.insert(i, next_i);
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
                &line_queue_item,
                &max_length,
                &hull_lines,
                &interior_points_tree,
                &current_hull_tree,
                &interior_lines_tree,
                &interior_polygon_tree,
            ) {
                let start_line = Line::new(line.start, candidate_point);
                let end_line = Line::new(candidate_point, line.end);

                if partial_min(Euclidean.length(&start_line), Euclidean.length(&end_line))
                    < max_length
                {
                    // Remove candidate point from interior points
                    remove_interior_point(&candidate_point, &mut interior_points_tree);

                    // Update current hull tree
                    current_hull_tree.remove(&line);
                    current_hull_tree.insert(start_line);
                    current_hull_tree.insert(end_line);

                    // Update hull lines
                    hull_lines.insert(line_queue_item.i, start_line);
                    hull_lines.insert(current_i, end_line);

                    // Either push line to queue or add confirmed line to hull order
                    if !is_interior_line(&start_line, &interior_lines_tree) {
                        line_queue.push_back(LineQueueItem {
                            line: start_line,
                            i: line_queue_item.i,
                            prev_i: line_queue_item.prev_i,
                            next_i: current_i,
                        });
                    } else {
                        hull_order.insert(line_queue_item.i, current_i);
                    }

                    if !is_interior_line(&end_line, &interior_lines_tree) {
                        line_queue.push_back(LineQueueItem {
                            line: end_line,
                            i: current_i,
                            prev_i: line_queue_item.i,
                            next_i: line_queue_item.next_i,
                        });
                    } else {
                        hull_order.insert(current_i, line_queue_item.next_i);
                    }

                    // Increment current_i for next new potential hull line
                    current_i += 1;
                    continue;
                }
            }
        }

        hull_order.insert(line_queue_item.i, line_queue_item.next_i);
    }
    Polygon::new(order_concave_hull(hull_order, hull_lines), vec![])
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
        ]
        .into();
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

    #[test]
    fn test_collinear_points() {
        let coords: Vec<Coord<f64>> = vec![
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 2.0, y: 0.0 },
            coord! { x: 2.0, y: 1.0 },
            coord! { x: 2.0, y: 2.0 },
            coord! { x: 0.0, y: 2.0 },
        ];
        let hull = coords.concave_hull(2.0, 0.0);
        let correct_hull = polygon![
            (x: 2.0, y: 0.0),
            (x: 2.0, y: 2.0),
            (x: 0.0, y: 2.0),
            (x: 0.0, y: 0.0),
            (x: 2.0, y: 0.0),
        ];
        assert_eq!(hull, correct_hull);
    }

    #[test]
    fn test_transverse_intersection() {
        let line_a = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 2.0, y: 2.0 });
        let line_b = Line::new(coord! { x: 0.0, y: 2.0 }, coord! { x: 2.0, y: 0.0 });
        assert!(transverse_intersection(&line_a, &line_b));

        let line_c = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 2.0, y: 0.0 });
        let line_d = Line::new(coord! { x: 1.0, y: 0.0 }, coord! { x: 1.0, y: 3.0 });
        assert!(!transverse_intersection(&line_c, &line_d));
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
