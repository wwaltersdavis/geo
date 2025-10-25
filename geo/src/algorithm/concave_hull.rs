use crate::bool_ops::BoolOpsNum;
use crate::convex_hull::qhull;
use crate::utils::partial_min;
use crate::{
    Buffer, Coord, CoordNum, Distance, Euclidean, GeoFloat, Geometry, GeometryCollection, Length, Line, LinesIter, LineString,
    Intersects, MultiLineString, MultiPoint, MultiPolygon, Polygon, coord, point,
};
use rstar::{Envelope, ParentNode, RTree, RTreeNode, RTreeNum, AABB};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, VecDeque},
    convert::TryInto,
};

pub trait ConcaveHull {
    type Scalar: CoordNum;
    fn concave_hull(&self, concavity: Option<Self::Scalar>, length_threshold: Option<Self::Scalar>) -> Polygon<Self::Scalar>;
}

impl<T> ConcaveHull for MultiPoint<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<T> {
        let mut coords: Vec<Coord<T>> = self.iter().map(|point| point.0).collect();
        concave_hull(&mut coords, concavity, length_threshold, None, None)
    }
}

impl<T> ConcaveHull for Polygon<T>
where
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut non_diggable_lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        add_polygon(self, &mut coords, &mut non_diggable_lines, &mut buffered_polygons);
        concave_hull(&mut coords, concavity, length_threshold, Some(non_diggable_lines), Some(buffered_polygons))
    }
}

impl<T> ConcaveHull for MultiPolygon<T>
where
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut non_diggable_lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        for polygon in self.0.iter() {
            add_polygon(polygon, &mut coords, &mut non_diggable_lines, &mut buffered_polygons);
        }
        concave_hull(&mut coords, concavity, length_threshold, Some(non_diggable_lines), Some(buffered_polygons))
    }
}

impl<T> ConcaveHull for LineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let non_diggable_lines: Vec<Line<T>> = self.lines().collect();
        concave_hull(&mut self.0.clone(), concavity, length_threshold, Some(non_diggable_lines), None)
    }
}

impl<T> ConcaveHull for MultiLineString<T>
where
    T: GeoFloat + RTreeNum,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<T> {
        let mut coords: Vec<Coord<T>> = self.iter().flat_map(|elem| elem.0.clone()).collect();
        let non_diggable_lines: Vec<Line<T>> = self.iter().flat_map(|elem| elem.lines()).collect();
        concave_hull(&mut coords, concavity, length_threshold, Some(non_diggable_lines), None)
    }
}

impl<T> ConcaveHull for GeometryCollection<T>
where
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let mut coords: Vec<Coord<T>> = Vec::new();
        let mut non_diggable_lines: Vec<Line<T>> = Vec::new();
        let mut buffered_polygons: Vec<Polygon<T>> = Vec::new();
        add_geometry_collection(self, &mut coords, &mut non_diggable_lines, &mut buffered_polygons);
        concave_hull(&mut coords, concavity, length_threshold, Some(non_diggable_lines), Some(buffered_polygons))
    }
}

fn add_geometry_collection<T>(
    collection: &GeometryCollection<T>,
    coords: &mut Vec<Coord<T>>,
    non_diggable_lines: &mut Vec<Line<T>>,
    buffered_polygons: &mut Vec<Polygon<T>>,
)
where
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
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
                non_diggable_lines.push(*line);
            }
            Geometry::LineString(line_string) => {
                coords.extend(line_string.0.iter());
                non_diggable_lines.extend(line_string.lines());
            }
            Geometry::MultiLineString(multi_line_string) => {
                for line_string in multi_line_string.iter() {
                    coords.extend(line_string.0.iter());
                    non_diggable_lines.extend(line_string.lines());
                }
            }
            Geometry::Polygon(polygon) => {
                add_polygon(polygon, coords, non_diggable_lines, buffered_polygons);
            }
            Geometry::MultiPolygon(multi_polygon) => {
                for polygon in multi_polygon.iter() {
                    add_polygon(polygon, coords, non_diggable_lines, buffered_polygons);
                }
            }
            Geometry::Triangle(triangle) => {
                let polygon: Polygon<T> = triangle.clone().try_into().unwrap();
                add_polygon(&polygon, coords, non_diggable_lines, buffered_polygons);
            }
            Geometry::Rect(rect) => {
                let polygon: Polygon<T> = rect.clone().try_into().unwrap();
                add_polygon(&polygon, coords, non_diggable_lines, buffered_polygons);
            }
            Geometry::GeometryCollection(nested_collection) => {
                add_geometry_collection(nested_collection, coords, non_diggable_lines, buffered_polygons);
            }
        }
    }
}

fn add_polygon<T>(
    polygon: &Polygon<T>,
    coords: &mut Vec<Coord<T>>,
    non_diggable_lines: &mut Vec<Line<T>>,
    buffered_polygons: &mut Vec<Polygon<T>>,
) 
where 
    T: GeoFloat + RTreeNum + BoolOpsNum + 'static,
{
    coords.extend(polygon.exterior().0.iter());
    non_diggable_lines.extend(polygon.exterior().lines_iter());
    buffered_polygons.extend(polygon.buffer(T::from(-1e-10).unwrap()).0);
}

fn line_diggable<T>(
    line: &Line<T>,
    non_diggable_lines_tree: &Option<RTree<Line<T>>>,
) -> bool
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = non_diggable_lines_tree {
        !tree.contains(line) && !tree.contains(&Line::new(line.end, line.start))
    } else {
        true
    }
}

enum RTreeNodePointer<'a, T> 
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
    tree_node: RTreeNodePointer<'a, T>,
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
        return T::zero();
    }
    let d2 = Euclidean.distance(line, &Line::new(c1, c2));
    if d2 == T::zero() {
        return T::zero();
    }
    let d3 = Euclidean.distance(line, &Line::new(c4, c3));
    if d3 == T::zero() {
        return T::zero();
    }
    let d4 = Euclidean.distance(line, &Line::new(c2, c3));
    if d4 == T::zero() {
        return T::zero();
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

// This is a rather hacky method of determining if a line is touching a geometry vs
// if it is intersecting it.
// TODO: Find a better way of doing this
fn shifted_intersects<T, G>(line: &Line<T>, inner_geometry: &G) -> bool 
where
    T: GeoFloat,
    G: Intersects<Line<T>>,
{
    // TODO: Take into account the length of the line and shift by a fraction of that
    let shift = T::from(1e-10).unwrap();

    inner_geometry.intersects(&Line::new(
        point!(x: line.start.x + shift, y: line.start.y + shift),
        point!(x: line.end.x + shift, y: line.end.y + shift),
    )) && 
    inner_geometry.intersects(&Line::new(
        point!(x: line.start.x - shift, y: line.start.y),
        point!(x: line.end.x - shift, y: line.end.y),
    )) && 
    inner_geometry.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y + shift),
        point!(x: line.end.x, y: line.end.y + shift),
    )) && 
    inner_geometry.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y - shift),
        point!(x: line.end.x, y: line.end.y - shift),
    ))
}

// TODO: Review if necessary
fn inner_line_intersections<T>(line: &Line<T>, tree: &Option<RTree<Line<T>>>) -> bool 
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = tree {
        let min_x = T::min(line.start.x, line.end.x);
        let max_x = T::max(line.start.x, line.end.x);
        let min_y = T::min(line.start.y, line.end.y);
        let max_y = T::max(line.start.y, line.end.y);
        let bbox = AABB::from_corners(point!([min_x, min_y]), point!([max_x, max_y]));
        let inner_lines = tree.locate_in_envelope_intersecting(&bbox);
        for inner_line in inner_lines {
            if shifted_intersects(line, inner_line) {
                return true;
            }
        }
    }
    false
}

fn no_inner_polygon_intersections<T>(line: &Line<T>, tree: &Option<RTree<Polygon<T>>>) -> bool 
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = tree {
        let min_x = T::min(line.start.x, line.end.x);
        let max_x = T::max(line.start.x, line.end.x);
        let min_y = T::min(line.start.y, line.end.y);
        let max_y = T::max(line.start.y, line.end.y);
        let bbox = AABB::from_corners(point!([min_x, min_y]), point!([max_x, max_y]));
        let inner_polygons = tree.locate_in_envelope_intersecting(&bbox);
        for inner_polygon in inner_polygons {
            if inner_polygon.intersects(line) {
                return false;
            }
        }
    }
    true
}

fn find_candidate<T>(
    interior_points_tree: &RTree<Coord<T>>,
    line: &Line<T>,
    current_hull_tree: &RTree<Line<T>>,
    max_length: &T,
    diggable_lines_tree: &RTree<Line<T>>,
    non_diggable_lines_tree: &Option<RTree<Line<T>>>,
    inner_polygon_tree: &Option<RTree<Polygon<T>>>,
) -> Option<Coord<T>>
where
    T: GeoFloat + RTreeNum,
{
    let mut queue: BinaryHeap<QueueItem<T>> = BinaryHeap::new();
    queue.push(QueueItem {
        tree_node: RTreeNodePointer::Parent(interior_points_tree.root()),
        distance: T::zero(),
    });

    while let Some(node) = queue.pop() {
        match node.tree_node {
            RTreeNodePointer::Parent(parent) => {
                for child in parent.children() {
                    match child {
                        RTreeNode::Parent(p) => {
                            let envelope = p.envelope();
                            let distance = line_to_bbox_distance(line, &envelope);
                            if distance <= *max_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodePointer::Parent(p),
                                    distance,
                                });
                            }
                        }
                        RTreeNode::Leaf(l) => {
                            let distance = Euclidean.distance(*l, line);
                            if distance <= *max_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodePointer::Leaf(l),
                                    distance,
                                });
                            }
                        }
                    }
                }
            }
            RTreeNodePointer::Leaf(leaf) => {
                // Ensure coord not the same as line endpoints
                if (*leaf == line.start) || (*leaf == line.end) {
                    continue;
                }
                // Ensure the potential candidate's point's nearest diggable line is the line in question
                if let Some(nearest_diggable_line) = diggable_lines_tree.nearest_neighbor(&point![x: leaf.x, y: leaf.y]) {
                    if nearest_diggable_line == line || nearest_diggable_line == &Line::new(line.end, line.start) {
                        let start_line = Line::new(line.start, *leaf);
                        let end_line = Line::new(*leaf, line.end);
                        if no_hull_intersections(&start_line, current_hull_tree) && 
                            no_hull_intersections(&end_line, current_hull_tree) &&
                            !inner_line_intersections(&start_line, non_diggable_lines_tree) && 
                            !inner_line_intersections(&end_line, non_diggable_lines_tree) &&
                            no_inner_polygon_intersections(&start_line, inner_polygon_tree) && 
                            no_inner_polygon_intersections(&end_line, inner_polygon_tree)
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

fn order_concave_hull<T>(concave_lines: HashMap<usize, (Line<T>, usize)>, convex_hull_length: usize) -> LineString<T>
where
    T: GeoFloat,
{
    let mut ordered_coords: Vec<Coord<T>> = vec![];
    let mut current_i = 0;
    ordered_coords.push(concave_lines.get(&current_i).unwrap().0.start);

    while current_i != convex_hull_length {
        if let Some((line, next_i)) = concave_lines.get(&current_i) {
            ordered_coords.push(line.end);
            current_i = *next_i;
        } else {
            break;
        }
    }
    LineString::from(ordered_coords)
}

struct LineQueueItem<T: GeoFloat + RTreeNum> {
    line: Line<T>,
    i: usize,
    next_i: usize,
}

fn concave_hull<T>(
    coords: &mut [Coord<T>],
    concavity: Option<T>, 
    length_threshold: Option<T>, 
    non_diggable_lines: Option<Vec<Line<T>>>,
    buffered_polygons: Option<Vec<Polygon<T>>>,
) -> Polygon<T>
where
    T: GeoFloat + RTreeNum,
{
    // Ensure concavity is non-negative, default to 2.0 if None
    let concavity: T = match concavity {Some(c) => T::max(T::zero(), c), None => T::from(2.0).unwrap()};
    let length_threshold: T = length_threshold.unwrap_or(T::from(0.0).unwrap());
    let hull = qhull::quick_hull(coords);

    // Index the points with an R-tree
    let mut interior_points_tree: RTree<Coord<T>> = RTree::bulk_load(coords.to_owned());
    let mut current_hull_tree: RTree<Line<T>> = RTree::bulk_load(hull.lines().collect());
    let non_diggable_lines_tree: Option<RTree<Line<T>>> = non_diggable_lines.map(RTree::bulk_load);
    let mut diggable_lines_tree: RTree<Line<T>> = RTree::new();

    let mut concave_lines: HashMap<usize, (Line<T>, usize)> = HashMap::new();
    let mut line_queue: VecDeque<LineQueueItem<T>> = VecDeque::new();
    for (i, line) in hull.lines().enumerate() {
        // line_queue.push_back(line);
        // Only populate queue with diggable lines
        if line_diggable(&line, &non_diggable_lines_tree) {
            diggable_lines_tree.insert(line);
            line_queue.push_back(LineQueueItem {line, i, next_i: i + 1});
        } else {
            concave_lines.insert(i, (line, i+1));
        }
        // Remove hull points from interior points
        if i == 0 {
            interior_points_tree.remove(&line.start);
        }
        interior_points_tree.remove(&line.end);
    }

    let buffered_polygon_tree: Option<RTree<Polygon<T>>> = buffered_polygons.map(RTree::bulk_load);

    let mut current_i = hull.0.len() + 1;
    while let Some(line_queue_item) = line_queue.pop_front() {
        let line = line_queue_item.line;
        let length = Euclidean.length(&line);
        if length >= length_threshold {
            let max_length = length / concavity;
            if let Some(candidate_point) = find_candidate(
                &interior_points_tree,
                &line,
                &current_hull_tree,
                &max_length,
                &diggable_lines_tree,
                &non_diggable_lines_tree,
                &buffered_polygon_tree) {
                    let start_line = Line::new(line.start, candidate_point);
                    let end_line = Line::new(candidate_point, line.end);
                    if partial_min(Euclidean.length(&start_line), Euclidean.length(&end_line)) < max_length {
                        // TODO: Handle removing potential duplicate points better
                        // Currently decreases performance by ~5%
                        let n = interior_points_tree.nearest_neighbors(&candidate_point).len();
                        for _ in 0..n {interior_points_tree.remove(&candidate_point);};
                        interior_points_tree.remove(&candidate_point);               
                        current_hull_tree.remove(&line);
                        current_hull_tree.insert(start_line);
                        current_hull_tree.insert(end_line);
                        diggable_lines_tree.remove(&line);
                        if line_diggable(&start_line, &non_diggable_lines_tree) {
                            diggable_lines_tree.insert(start_line);
                            line_queue.push_back(LineQueueItem {
                                line: start_line,
                                i: line_queue_item.i,
                                next_i: current_i,
                            });
                        } else {
                            concave_lines.insert(line_queue_item.i, (start_line, current_i));
                        }
                        if line_diggable(&end_line, &non_diggable_lines_tree) {
                            diggable_lines_tree.insert(end_line);
                            line_queue.push_back(LineQueueItem {
                                line: end_line,
                                i: current_i,
                                next_i: line_queue_item.next_i,
                            });
                        } else {
                            concave_lines.insert(current_i, (end_line, line_queue_item.next_i));
                        }
                        current_i += 1;
                        continue;
                    }
                }
        }
        concave_lines.insert(line_queue_item.i, (line, line_queue_item.next_i));
    }
    Polygon::new(order_concave_hull(concave_lines, hull.0.len()), vec![])
}