import { Course } from "types/course";

export const normalizeCourses = (coursesFromApi: any[]): Course[] => {
  return coursesFromApi.map((course) => ({
    id: course.id,
    title: course.title,
    duration: course.duration,
    description: course.description,
    url: course.url,
    imageUrl: course.image_url,
  }));
};
