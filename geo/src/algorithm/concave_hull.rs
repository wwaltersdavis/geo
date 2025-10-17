use crate::convex_hull::qhull;
use crate::utils::partial_min;
use crate::{
    Coord, CoordNum, Distance, Euclidean, GeoFloat, Length, Line, LinesIter, LineString,
    Intersects, MultiLineString, MultiPoint, MultiPolygon, Polygon, coord, point,
};
use rstar::{Envelope, ParentNode, RTree, RTreeNode, RTreeNum, AABB};
use core::panic;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::fmt::Display;

pub trait ConcaveHull {
    type Scalar: CoordNum;
    fn concave_hull(&self, concavity: Option<Self::Scalar>, length_threshold: Option<Self::Scalar>) -> Polygon<Self::Scalar>;
}

impl<T> ConcaveHull for Polygon<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let mut points: Vec<Coord<T>> = self.exterior().0.clone();
        let polygons = vec![self.clone()];
        // TODO: Just get the exterior line of the polygon (as could be an issue with intersects)
        concave_hull(&mut points, concavity, length_threshold, Some(polygons), None)
    }
}

impl<T> ConcaveHull for MultiPolygon<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let mut aggregated: Vec<Coord<T>> = self
            .0
            .iter()
            .flat_map(|elem| elem.exterior().0.clone())
            .collect();
        let polygons: Vec<Polygon<T>> = self.0.clone();
        concave_hull(&mut aggregated, concavity, length_threshold, Some(polygons), None)
    }
}

impl<T> ConcaveHull for LineString<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<Self::Scalar> {
        let lines: Vec<Line<T>> = self.lines().collect();
        concave_hull(&mut self.0.clone(), concavity, length_threshold, None, Some(lines))
    }
}

impl<T> ConcaveHull for MultiLineString<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<T> {
        let mut aggregated: Vec<Coord<T>> = self.iter().flat_map(|elem| elem.0.clone()).collect();
        let lines: Vec<Line<T>> = self.iter().flat_map(|elem| elem.lines()).collect();
        concave_hull(&mut aggregated, concavity, length_threshold, None, Some(lines))
    }
}

impl<T> ConcaveHull for MultiPoint<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    type Scalar = T;
    fn concave_hull(&self, concavity: Option<T>, length_threshold: Option<T>) -> Polygon<T> {
        let mut coordinates: Vec<Coord<T>> = self.iter().map(|point| point.0).collect();
        concave_hull(&mut coordinates, concavity, length_threshold, None, None)
    }
}

fn non_diggable_lines<T>(
    polygons: &Option<Vec<Polygon<T>>>,
    lines: &Option<Vec<Line<T>>>,
) -> Option<RTree<Line<T>>> 
where 
    T: GeoFloat + RTreeNum,
{
    let mut all_lines = Vec::new();
    
    if let Some(polygons) = polygons {
        all_lines.extend(polygons.iter().flat_map(|polygon| polygon.lines_iter()));
    }

    if let Some(lines) = lines {
        all_lines.extend(lines.iter().copied());
    }
    
    if all_lines.is_empty() {
        None
    } else {
        Some(RTree::bulk_load(all_lines))
    }
}

fn line_not_diggable<T>(
    line: &Line<T>,
    tree: &Option<RTree<Line<T>>>,
) -> bool
where
    T: GeoFloat + RTreeNum,
{
    if let Some(tree) = tree {
        !tree.contains(line) && !tree.contains(&Line::new(line.end, line.start))
    } else {
        true
    }
}

fn get_square_distance<T>(line: &Line<T>) -> T 
where
    T: GeoFloat,
{
    let length = Euclidean.length(line);
    length * length
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

fn square_line_to_bbox_distance<T>(line: &Line<T>, aabb: &AABB<Coord<T>>) -> T
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
    let d1 = Euclidean.distance(line, &Line::new(c1, c4)).powi(2);
    if d1 == T::zero() {
        return T::zero();
    }
    let d2 = Euclidean.distance(line, &Line::new(c1, c2)).powi(2);
    if d2 == T::zero() {
        return T::zero();
    }
    let d3 = Euclidean.distance(line, &Line::new(c4, c3)).powi(2);
    if d3 == T::zero() {
        return T::zero();
    }
    let d4 = Euclidean.distance(line, &Line::new(c2, c3)).powi(2);
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
fn shifted_intersects<T>(line: &Line<T>, inner_line: &Line<T>) -> bool 
where
    T: GeoFloat,
{
    // TODO: Take into account the length of the line and shift by a fraction of that
    let shift = T::from(1e-10).unwrap();

    if !inner_line.intersects(&Line::new(
        point!(x: line.start.x + shift, y: line.start.y + shift),
        point!(x: line.end.x + shift, y: line.end.y + shift),
    )) {
        return false;
    }
    if !inner_line.intersects(&Line::new(
        point!(x: line.start.x - shift, y: line.start.y),
        point!(x: line.end.x - shift, y: line.end.y),
    )) {
        return false;
    }
    if !inner_line.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y + shift),
        point!(x: line.end.x, y: line.end.y + shift),
    )) {
        return false;
    }
    if !inner_line.intersects(&Line::new(
        point!(x: line.start.x, y: line.start.y - shift),
        point!(x: line.end.x, y: line.end.y - shift),
    )) {
        return false;
    }
    true
}

// TODO: Review if necessary
fn no_inner_line_intersections<T>(line: &Line<T>, tree: &Option<RTree<Line<T>>>) -> bool 
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
    max_square_length: &T,
    diggable_lines_tree: &RTree<Line<T>>,
    non_diggable_lines_tree: &Option<RTree<Line<T>>>,
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
                            // TODO: use Geo's internal functions here and test performance
                            let square_distance = square_line_to_bbox_distance(line, &envelope);
                            if square_distance <= *max_square_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodePointer::Parent(p),
                                    distance: square_distance,
                                });
                            }
                        }
                        RTreeNode::Leaf(l) => {
                            let square_distance = Euclidean.distance(*l, line).powi(2);
                            if square_distance <= *max_square_length {
                                queue.push(QueueItem {
                                    tree_node: RTreeNodePointer::Leaf(l),
                                    distance: square_distance,
                                });
                            }
                        }
                    }
                }
            }
            RTreeNodePointer::Leaf(leaf) => {
                // Ensure the potential candidate's point's nearest line is the line in question
                if let Some(nearest_diggable_line) = diggable_lines_tree.nearest_neighbor(&point![x: leaf.x, y: leaf.y]) {
                    if nearest_diggable_line == line || nearest_diggable_line == &Line::new(line.end, line.start) {
                        let line1 = Line::new(line.start, *leaf);
                        let line2 = Line::new(line.end, *leaf);
                        if no_hull_intersections(&line1, current_hull_tree) && no_hull_intersections(&line2, current_hull_tree) &&
                           no_inner_line_intersections(&line1, non_diggable_lines_tree) && no_inner_line_intersections(&line2, non_diggable_lines_tree)
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

// TODO: Remove this function along with the Display requirement after testing
fn coord_as_string<T>(coord: &Coord<T>) -> String
where
    T: GeoFloat + Display,
{
    format!("{}_{}", coord.x, coord.y)
}

fn order_concave_hull<T>(lines: Vec<Line<T>>) -> LineString<T>
where
    T: GeoFloat + Display,
{
    // Assume all coordinates are unique
    let mut line_map: HashMap<String, Coord<T>> = HashMap::new();
    for line in &lines {
        line_map.insert(coord_as_string(&line.start), line.end);
    }
    let mut ordered_coords: Vec<Coord<T>> = vec![];
    // Start from an arbitrary line
    for _ in 0..lines.len() {
        if ordered_coords.is_empty() {
            let first_line = &lines[0];
            ordered_coords.push(first_line.start);
            ordered_coords.push(first_line.end);
        } else {
            let last_coord = ordered_coords.last().unwrap();
            if let Some(next_coord) = line_map.get(&coord_as_string(last_coord)) {
                ordered_coords.push(*next_coord);
            } else {
                // TODO: Remove this after testing
                panic!("Could not find coord after {:?}", last_coord);
            }
        }
    }
    LineString::from(ordered_coords)
}

fn concave_hull<T>(
    coords: &mut [Coord<T>],
    concavity: Option<T>, 
    length_threshold: Option<T>, 
    polygons: Option<Vec<Polygon<T>>>, 
    inner_lines: Option<Vec<Line<T>>>
) -> Polygon<T>
where
    T: GeoFloat + RTreeNum + Display,
{
    // Ensure concavity is non-negative, default to 2.0 if None
    let concavity: T = match concavity {Some(c) => T::max(T::zero(), c), None => T::from(2.0).unwrap()};
    let length_threshold: T = length_threshold.unwrap_or(T::from(0.0).unwrap());
    let hull = qhull::quick_hull(coords);

    // TODO: ensure all coords are unique

    // Index the points with an R-tree
    let mut interior_points_tree: RTree<Coord<T>> = RTree::bulk_load(coords.to_owned());
    let mut current_hull_tree: RTree<Line<T>> = RTree::bulk_load(hull.lines().collect());
    let non_diggable_lines_tree: Option<RTree<Line<T>>> = non_diggable_lines(&polygons, &inner_lines);
    let mut diggable_lines_tree: RTree<Line<T>> = RTree::new();
    let mut unordered_concave_lines: Vec<Line<T>> = vec![];

    let mut line_queue: VecDeque<Line<T>> = VecDeque::new();
    for (i, line) in hull.lines().enumerate() {
        // line_queue.push_back(line);
        // Only populate queue with diggable lines
        if line_not_diggable(&line, &non_diggable_lines_tree) {
            diggable_lines_tree.insert(line);
            line_queue.push_back(line);
        } else {
            unordered_concave_lines.push(line);
        }
        // Remove hull points from interior points
        // TODO: Fix this hacky way of ensuring we only remove each hull point once
        if i == 0 {
            interior_points_tree.remove(&line.start);
        }
        interior_points_tree.remove(&line.end);
    }

    // let inner_polygon_tree: Option<RTree<Polygon<T>>> = polygons.map(RTree::bulk_load);
    // TODO: Remove interior points that are within inner polygons

    let square_concavity = concavity * concavity;
    let square_length_threshold = length_threshold * length_threshold;

     while let Some(line) = line_queue.pop_front() {
        if line_not_diggable(&line, &non_diggable_lines_tree) {
            continue;
        }

        let square_length = get_square_distance(&line);
        if square_length >= square_length_threshold {
            let max_square_length = square_length / square_concavity;
            if let Some(candidate_point) = find_candidate(
                &interior_points_tree, 
                &line,
                &current_hull_tree,
                &max_square_length, 
                &diggable_lines_tree, 
                &non_diggable_lines_tree) {
                    let start_line = Line::new(line.start, candidate_point);
                    let end_line = Line::new(candidate_point, line.end);
                    if partial_min(get_square_distance(&start_line), get_square_distance(&end_line)) < max_square_length {
                        interior_points_tree.remove(&candidate_point);
                        current_hull_tree.remove(&line);
                        current_hull_tree.insert(start_line);
                        current_hull_tree.insert(end_line);
                        diggable_lines_tree.remove(&line);
                        line_queue.push_back(start_line);
                        line_queue.push_back(end_line);
                        continue;
                    }
                }
        }
        unordered_concave_lines.push(line);
     }
    Polygon::new(order_concave_hull(unordered_concave_lines), vec![])
}